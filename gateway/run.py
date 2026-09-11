"""Gateway runner - entry point for messaging platform integrations.

Provides ``start_gateway()`` (start all configured adapters) and ``GatewayRunner`` (lifecycle).
Run via ``python -m gateway.run`` or ``python cli.py --gateway``."""

# hermes_bootstrap must be the very first import (UTF-8 stdio on Windows; no-op on POSIX).
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass  # a partial ``hermes update`` can leave the bootstrap unregistered; only Windows UTF-8 stdio suffers

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import shlex
import site
import sys
import signal
import threading
import time
import traceback
from collections import OrderedDict
from contextvars import copy_context
from pathlib import Path
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional, Any, List, Tuple, Union, cast

from agent.async_utils import safe_schedule_threadsafe
from agent.conversation_compression import (
    COMPACTION_DONE_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_STATUS, COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE, COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
    PRE_API_COMPRESSION_STATUS_TEMPLATE, PREFLIGHT_COMPRESSION_STATUS_TEMPLATE)
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from agent.interrupt_compat import request_hard_interrupt
from agent.turn_context import compression_made_progress
from agent.session_activity import ActivityProvenance
from hermes_cli.config import _is_ssh_remote_tilde_cwd, cfg_get
from hermes_cli.fallback_config import get_fallback_chain

# Per-session AIAgent cache bounds (agents are heavy); see _enforce_agent_cache_cap/_session_housekeeping_watcher.
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
# Telegram connect proves a real getUpdates round trip; must cover polling-start deadlines + readiness.
_TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT = 180.0
# The initial Telegram connect gates `running` for EVERY platform, so it must not spend the full 180s.
# Cold-start cap for Telegram (#85993): the initial connect awaited before the gateway reaches `running`
# must not spend the full 180s budget — an unreachable Telegram would hold EVERY platform's serving state
# hostage for the whole window. The initial attempt gets one bounded try; on timeout the platform is queued
# for the reconnect watcher, which retries with the full 180s budget (is_reconnect=True preserves the
# offline update queue, #46621).
_TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT = 45.0
_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
# End reasons meaning the USER deliberately closed this thread. Shared by _classify_completion_target and
# _resolve_async_delegation_session so they never disagree (else a "delivered" reason is acked, then lost).
_USER_BOUNDARY_END_REASONS = ("session_reset", "user_exit", "session_switch", "new_session")
# Bounds one stall-notify send so a wedged transport can't block the watcher; on timeout the next tick retries.
_STALL_NOTIFY_SEND_TIMEOUT_SECONDS = 15.0
_GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS = 16 * 1024 * 1024
_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")
_GATEWAY_HYGIENE_PLATFORM = "gateway_hygiene"

_TELEGRAM_NOISY_STATUS_RE = re.compile(
    r"("  # transient/auxiliary status that should stay in logs, not gateway chats
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    # the auto-lower notice was reworded to "Auto-lowered this session's threshold..." — cover both.
    # See #69332.
    r"|auto-lowered\s+(?:this\s+)?session'?s?\s+threshold"
    r"|configured\s+auxiliary\s+compression\s+provider\s+.+\s+unavailable"
    r"|skipping\s+concurrent\s+compression"
    rf"|{re.escape(COMPACTION_STATUS)}"
    rf"|{re.escape(COMPACTION_HEARTBEAT_STATUS)}"
    r"|resumed\s+after\s+\d+s\s+idle\s+[—-]\s+compacting"
    r"|preflight\s+compression"
    r"|pre[- ]api\s+compression"
    # Retry chatter via _emit_status; ", retrying"/"— compressing" anchors exclude manual /compress feedback.
    r"|context\s+too\s+large\s+\(~[\d,]+\s+tokens\)\s+[—-]+\s+compressing"
    r"|compressed\s+\d[\d,]*\s+(?:→|->)\s+\d[\d,]*\s+messages,\s+retrying"
    r"|compressed\s+~[\d,]+\s+(?:→|->)\s+~[\d,]+\s+tokens,\s+retrying"
    r"|context\s+reduced\s+to\s+[\d,]+\s+tokens\s+\(was\s+[\d,]+\),\s+retrying"
    r"|session\s+compressed\s+\d+\s+times"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    rf"|{re.escape(COMPACTION_DONE_STATUS)}"
    r")",
    re.IGNORECASE | re.DOTALL)

_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS = (1, 3, 9)
# Ceiling on an escalated cooldown (cf. _RECONNECT_BACKOFF_CAP): base × ladder can reach 9h ≈ "compaction off".
_HYGIENE_COOLDOWN_MAX_SECONDS = 3600.0
# Flat retry-after when hygiene is ABANDONED by turn-hold expiry (not a failure: outside the streak ladder).
_HYGIENE_TURNHOLD_RETRY_SECONDS = 60.0


def _gateway_session_db_inner(gateway):
    """The raw SessionDB behind ``gateway._session_db`` (unwrapping the async facade), or None."""
    session_db = getattr(gateway, "_session_db", None)
    return getattr(session_db, "_db", session_db)


def _hygiene_cooldown_for_failure(gateway, session_key: str, base_cooldown_seconds: float) -> float:
    """Bump the hygiene failure streak and return the escalated cooldown (x1/x3/x9 over base, clamped).

    Hygiene's per-run ``AIAgent`` is fresh, so the streak lives in SQLite keyed by rotation-stable session_key.

    It exists because the in-agent equivalent is unreachable from here:
    ``ContextCompressor.record_timeout_failure`` escalates on an absolute 60 -> 300 -> 900s ladder driven by
    the in-memory ``_consecutive_timeout_failures`` counter, which ``bind_session_state`` zeroes. Session
    hygiene constructs a FRESH ``AIAgent`` per run and re-binds state every time, so from the gateway that
    streak is structurally always 0 and only the flat ``hygiene_failure_cooldown_seconds`` could ever be
    recorded — a session whose summary model always times out retried on that same fixed interval forever
    (#79624). The streak is mirrored to SQLite by rotation-stable ``session_key`` so it outlives both the
    per-run agent and gateway restarts; ``PersistentState`` keeps the hot in-process view.
    """
    streak, state = 1, None
    try:
        state = gateway._session_state(session_key).persistent
    except Exception as exc:
        logger.debug("hygiene failure streak update failed: %s", exc)
    increment = getattr(_gateway_session_db_inner(gateway), "increment_hygiene_failure_streak", None)
    if callable(increment):
        try:
            streak = max(1, int(increment(session_key)))
            if state is not None:
                state.hygiene_failure_streak = streak
        except Exception as exc:
            logger.debug("hygiene failure streak persist failed: %s", exc)
            if state is not None:
                state.hygiene_failure_streak += 1
                streak = state.hygiene_failure_streak
    elif state is not None:
        state.hygiene_failure_streak += 1
        streak = state.hygiene_failure_streak
    multiplier = _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS[
        min(streak, len(_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS)) - 1]
    return min(base_cooldown_seconds * multiplier, _HYGIENE_COOLDOWN_MAX_SECONDS)


def _reset_hygiene_failure_streak(gateway, session_key: str) -> None:
    """Clear the hygiene failure streak after a compression that reduced context.

    Peeks, never get-or-creates: a no-op 0 write must not create a never-evicted ``_sessions`` row."""
    try:
        state = gateway._peek_session_state(session_key)
        if state is not None:
            state.persistent.hygiene_failure_streak = 0
    except Exception as exc:
        logger.debug("hygiene failure streak reset failed: %s", exc)
    reset = getattr(_gateway_session_db_inner(gateway), "reset_hygiene_failure_streak", None)
    if callable(reset):
        try:
            reset(session_key)
        except Exception as exc:
            logger.debug("hygiene failure streak persistent reset failed: %s", exc)


def hygiene_compaction_recovered(
    *, aborted: bool, rotated: bool, in_place: bool, msg_count: int, new_count: int,
    approx_tokens: int, new_tokens: int) -> bool:
    """True when a hygiene run actually recovered the session (extracted to be unit testable).

    Requires no abort, a real rewrite (the no-op path reuses pre-compression counts) and material shrink per
    :func:`compression_made_progress` (a bare ``<`` misses row-count wins and counts estimate noise).

    * the compressor did not abort (no summary produced at all); * the transcript was actually rewritten —
    either rotated into a new session or compacted in place. The degenerate "did not rotate or compact in
    place" path (#21301) reuses the pre-compression counts, so relying on the numbers alone would read a
    no-op as success; * the request materially shrank, per the canonical :func:`compression_made_progress`
    (#39548) — a row-count drop counts even when the summary keeps the token estimate flat, and a sub-5%
    token wobble does not count at all.
    """
    if aborted or not (rotated or in_place):
        return False
    return compression_made_progress(msg_count, new_count, approx_tokens, new_tokens)


def _hygiene_compression_timeout_message(
    *, total_exhausted: bool, elapsed: float, idle_timeout: float, progress_observed: bool) -> str:
    """Describe the host timeout that actually ended hygiene compression."""
    if total_exhausted:
        progress = " after summary output was observed" if progress_observed else ""
        return (
            "⚠️ Context compression reached its total ceiling after "
            f"{elapsed:.1f}s{progress}. No messages were dropped — continuing "
            "without compression. Run /compress to retry or /reset for a clean session.")
    return (
        f"⚠️ Context compression timed out after {idle_timeout:.1f}s with no "
        "output from the summary model. No messages were dropped — continuing "
        "without compression. Run /compress to retry, /reset for a clean "
        "session, or check your auxiliary.compression model configuration.")


def _cached_agent_for_hygiene(gateway, session_key: str):
    """The cached live AIAgent for ``session_key`` (or the pending sentinel / None), read under the cache lock."""
    cache = getattr(gateway, "_agent_cache", None)
    if cache is None:
        return None
    lock = getattr(gateway, "_agent_cache_lock", None)
    try:
        with (lock or suppress()):
            entry = cache.get(session_key)
    except Exception:
        entry = None
    return entry[0] if isinstance(entry, tuple) and entry else entry


async def run_codex_hygiene_compaction(
    gateway, session_key: str, session_id: str, *, auto_mode: str, history: list,
    approx_tokens: int, timeout_seconds: float, failure_cooldown_seconds: float = 300.0) -> str:
    """Session hygiene for ``codex_app_server`` sessions.

    The real context is the server-side thread; the local transcript is a never-replayed mirror, so rewriting
    it shrinks nothing and evicting the live agent starts the next turn on an EMPTY thread. So: compact the LIVE
    agent via ``thread/compact/start``, keep it cached, never build a detached compressor. ``native``/``off``
    skip without local fallback. Returns ``compacted``, ``skipped:<reason>`` or ``failed:<reason>``.

    See #73503.
    * Evicting the cached live agent afterwards destroys the only real context: the next turn spawns an
    EMPTY thread and the model starts blank while Hermes still mirrors a full history (abrupt amnesia — the
    user-facing damage documented on #73503).
    """
    mode = str(auto_mode or "native").lower()
    if mode not in {"native", "hermes", "off"}:
        mode = "native"
    if mode != "hermes":
        # native = app-server compacts itself; off = operator disabled. Local fallback can't shrink the thread.
        return f"skipped:mode={mode}"

    agent = _cached_agent_for_hygiene(gateway, session_key)
    if agent is None or agent is _AGENT_PENDING_SENTINEL:
        # No live agent → no live thread; a detached mirror-only rewrite is the no-op this exists to remove.
        return "skipped:no-cached-agent"
    if getattr(agent, "_codex_session", None) is None:
        return "skipped:no-live-thread"

    compressor = getattr(agent, "context_compressor", None)
    count_before = getattr(compressor, "compression_count", 0)
    # copy_context carries profile secret scope / HERMES_HOME override (executors don't propagate ContextVars).
    worker_future = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, lambda: agent._compress_context(history, "", approx_tokens=approx_tokens))
    track_worker = getattr(gateway, "_track_deferred_agent_worker", None)
    if callable(track_worker):
        # ``wait_for`` only cancels the asyncio wrapper; keep the running executor thread visible to shutdown.
        track_worker(worker_future, agent)
    try:
        await asyncio.wait_for(asyncio.shield(worker_future), timeout=max(float(timeout_seconds), 1.0))
    except asyncio.TimeoutError:
        # Executor thread keeps running (own RPC timeouts); brake retries so a wedged app-server isn't re-hit.
        if failure_cooldown_seconds >= 0:
            _record_hygiene_cooldown(
                gateway, session_id, failure_cooldown_seconds, "codex app-server thread compaction timed out")
        logger.warning(
            "Session hygiene: codex app-server thread compaction for "
            "session %s timed out after %.1fs; continuing without compaction",
            session_id, timeout_seconds)
        return "failed:timeout"
    except Exception as exc:
        logger.warning(
            "Session hygiene: codex app-server thread compaction for session %s failed: %s", session_id, exc)
        return f"failed:{exc}"

    count_after = getattr(compressor, "compression_count", 0)
    if count_after > count_before:
        # Native boundary recorded: compacted server-side; mirror NOT rewritten, agent stays cached.
        _reset_hygiene_failure_streak(gateway, session_key)
        return "compacted"
    # No boundary: internal skip or compaction error; the codex route already persisted its own cooldown.
    return "failed:no-boundary"

def hygiene_wait_should_extend(
    *, idle: float, timeout: float, waited: float, ceiling: float, fence_cancelled: bool = False
) -> bool:
    """Whether the hygiene host should keep waiting for a slow summary.

    A cancelled commit fence cannot commit: extending only queues inbound messages behind a doomed attempt.

    Stop extending immediately so the turn can continue. See #96953.
    """
    return not fence_cancelled and idle < timeout and waited < ceiling


def _record_hygiene_cooldown(
    gateway, session_id: str, cooldown_seconds: float, error: Optional[str] = None) -> None:
    """Persist a session-hygiene compression-failure cooldown to the state DB (survives restarts).

    ``error`` must be forwarded: the recorder writes compression_failure_error UNCONDITIONALLY (NULL clobber).

    Uses the same ``compression_failure_cooldown_until`` column and ``record_compression_failure_cooldown``
    method that the in-conversation compression path (``agent/context_compressor.py``) already uses, so the
    cooldown survives gateway restarts (#74136).
    """
    recorder = getattr(_gateway_session_db_inner(gateway), "record_compression_failure_cooldown", None)
    if recorder is None:
        return
    try:
        recorder(session_id, time.time() + cooldown_seconds, error)
    except Exception as exc:
        logger.debug("session hygiene cooldown persist failed: %s", exc)


def _status_template_to_regex(template: str) -> str:
    """Compile a compression status template constant into a regex source.

    Literal text is escaped verbatim (wording drift can't diverge from the matcher); ``{field}`` -> numeric."""
    parts = re.split(r"\{[^{}]*\}", template)
    return r"[\d,]+".join(re.escape(part) for part in parts)


# ROUTINE compression progress statuses, derived from the SAME template constants the emit sites format.
# Used ONLY by the opt-in ``compression.progress_notices`` gate below (#52995) to decide which of the noisy
# statuses matched by _TELEGRAM_NOISY_STATUS_RE are compression progress (deliverable when the user opted
# in) versus unrelated aux/retry chatter (always suppressed on chat surfaces). Failure notices and manual
# /compress feedback never match _TELEGRAM_NOISY_STATUS_RE in the first place, so they are unaffected by
# this gate.
_COMPRESSION_PROGRESS_STATUS_RE = re.compile(
    "|".join(
        _status_template_to_regex(_template)
        for _template in (
            COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_DONE_STATUS, PRE_API_COMPRESSION_STATUS_TEMPLATE,
            PREFLIGHT_COMPRESSION_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
            COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE)),
    re.IGNORECASE)


def _gateway_compression_progress_notices_enabled() -> bool:
    """True when ``compression.progress_notices`` is on (default False: chat is silent by design).

    Read live (mtime-cached) so a config edit applies at the next status; fail-closed on read error.

    Reads ``compression.progress_notices`` from the gateway's raw YAML config (#52995).
    """
    try:
        config = _load_gateway_config()
        compression_cfg = config.get("compression") if isinstance(config, dict) else None
        if isinstance(compression_cfg, dict):
            return str(compression_cfg.get("progress_notices", False)).strip().lower() in {
                "true", "1", "yes", "on"}
    except Exception:
        pass
    return False

# Surfaces consuming gateway text programmatically must keep RAW status/error text; unknown/empty -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset({"local", "api_server", "webhook", "msgraph_webhook"})


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


_GATEWAY_PROVIDER_POLICY_RE = re.compile(
    r"("  # raw provider policy/safety bodies are noisy and may be sensitive
    r"cybersecurity\s+risk"
    r"|security\s+policy"
    r"|safety\s+policy"
    r"|policy\s+violation"
    r"|violat(?:e|es|ed|ion)"
    r"|blocked\s+(?:because|by|under)"
    r"|request\s+(?:was\s+)?(?:blocked|rejected)"
    r"|disallowed"
    r"|moderation"
    r")",
    re.IGNORECASE)

_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key|\b401\b)",
    re.IGNORECASE)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)", re.IGNORECASE)

# Connection-failure markers: the first 8 also anchor the provider-failure envelope shape below.
_CONNECTION_ERROR_MARKERS = (
    r"(?:\w+\.)?(?:api\s*)?connection\s*(?:error|timeout)", r"(?:\w+\.)?connect\s*(?:error|timeout)",
    r"connection\s+refused", r"connection\s+reset", r"connection\s+aborted", r"actively\s+refused",
    r"winerror\s+10061", r"errno\s+111", r"no\s+route\s+to\s+host", r"network\s+is\s+unreachable",
    r"cannot\s+connect", r"failed\s+to\s+establish", r"could\s+not\s+connect")
_GATEWAY_CONNECTION_ERROR_RE = re.compile("(" + "|".join(_CONNECTION_ERROR_MARKERS) + ")", re.IGNORECASE)

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), re.compile(r"\bxapp-\d+-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"), re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"))


def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Patched before MCP discovery so tool injection does not depend on launchers preserving PYTHONPATH."""
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(project_root / "venv")

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue

        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitedir semantics matter: pywin32 (MCP SDK on Windows) needs .pth processing for pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        pythonpath = [project_entry, site_entry]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        return


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _non_conversational_metadata(
    metadata: Optional[Dict[str, Any]] = None, *, platform: Any = None) -> Optional[Dict[str, Any]]:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if _gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


def _interim_metadata(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mark a mid-turn status/advisory send as NOT the turn-final.

    Stream-is-the-message adapters seal the live stream with the first unmarked send to an armed (chat, turn)
    key, so every mid-turn send MUST carry this marker. Gateway-internal; adapters strip it before the wire."""
    merged = dict(metadata or {})
    merged["_interim_send"] = True
    return merged


def _seed_hygiene_system_prompt(agent: Any, session_row: Optional[Dict[str, Any]]) -> bool:
    """Keep gateway hygiene from rebuilding a live session's system prompt.

    Hygiene lacks the live prompt environment, so a rebuild (persisted by compression) would strip external
    provider blocks. Seed the persisted prompt (or an empty cache entry); the real turn rebuilds properly."""
    stored_prompt = ""
    if isinstance(session_row, dict):
        raw_prompt = session_row.get("system_prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            stored_prompt = raw_prompt

    agent._cached_system_prompt = stored_prompt
    return bool(stored_prompt)


_TRANSIENT_NETWORK_ERROR_CLASS_NAMES = frozenset({
    "TimedOut", "NetworkError", "ReadError", "WriteError", "ConnectError", "ConnectTimeout",
    "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "ServerDisconnectedError",
    "ClientConnectorError", "ClientOSError"})


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for transient network errors safe to log + swallow (the next poll recovers; never crash).

    Walks the cause chain so wrapped errors (PTB ``NetworkError`` over ``httpx.ConnectError``) match.

    The crash class targeted by #31066 / #31110: an unhandled Telegram ``TimedOut`` (or peer
    ``NetworkError`` / ``httpx`` connection error) propagating to the event loop and killing the entire
    gateway process. These are by definition transient — the next poll cycle or user action recovers — so
    they must never crash the process.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 12:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        if type(cur).__name__ in _TRANSIENT_NETWORK_ERROR_CLASS_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _gateway_loop_exception_handler(
    loop: "asyncio.AbstractEventLoop", context: Dict[str, Any]) -> None:
    """Loop-level safety net for transient network errors (installed once by ``start_gateway``).

    Logs WARNING with traceback; non-transient errors go to the default handler so real bugs surface.

    Catches the ``telegram.error.TimedOut`` crash class (issues #31066 / #31110) and any peer transient
    network error before it can kill the gateway process.
    """
    exc = context.get("exception")
    if exc is not None and _is_transient_network_error(exc):
        task = context.get("future") or context.get("task")
        task_name = ""
        if task is not None:
            try:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
            except Exception:
                task_name = repr(task)
        logger.warning(
            "Gateway swallowed transient network error from %s: %s: %s", task_name or "<unknown task>",
            type(exc).__name__, exc, exc_info=(type(exc), exc, exc.__traceback__))
        return
    loop.default_exception_handler(context)


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Secret redaction before text can leave the gateway.

    Shared ``redact_sensitive_text`` with ``force=True`` (holds even when ``security.redact_secrets`` is off);
    ``_GATEWAY_SECRET_PATTERNS`` is a second pass so redaction degrades gracefully if that import fails.

    Delegates to the authoritative ``agent.redact.redact_sensitive_text`` — the same Tirith-grade redactor
    already applied to logs, tool output, and approval-command prompts — so the outbound chat path masks the
    full credential set the startup banner promises ("chat responses are scrubbed before delivery"), not a
    divergent subset. See #23810.
    """
    redacted = str(text or "")
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(redacted, force=True)
    except Exception:
        pass  # fail-soft: the local pattern pass below still runs rather than leaking raw text to chat
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _redact_approval_command(cmd: "str | None") -> str:
    """Redact credentials from a command before it goes into an approval prompt.

    Else a Tirith-flagged credential echoes verbatim to chat; ``force=True`` holds even with redaction off.

    Tirith's *findings* are already redacted, but the gateway approval prompt is built from the raw command
    string, so a credential-shaped value Tirith flagged would otherwise be echoed verbatim to the chat
    platform (#48456). Uses ``redact_sensitive_text(force=True)`` — the same Tirith-grade redactor — so the
    prompt honors redaction even when ``security.redact_secrets`` is off. Module-level so the wiring is
    unit-testable (the call site is a deeply nested gateway closure that cannot be driven directly).
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(cmd or ""), force=True)


def _format_exec_approval_fallback(
    command: str, description: str, command_prefix: str, *, allow_permanent: bool = True,
    allow_session: bool = True, smart_denied: bool = False) -> str:
    """Render the text fallback from approval capabilities, not platform names."""
    cmd_preview = command[:200] + "..." if len(command) > 200 else command
    heading = ("⚠️ **Smart DENY — owner override for one operation:**" if smart_denied
               else "⚠️ **Dangerous command requires approval:**")

    choices = [f"Reply `{command_prefix}approve` to execute this one operation"]
    if not smart_denied and allow_session:
        choices.append(f"`{command_prefix}approve session` to approve this pattern for the session")
        if allow_permanent:
            choices.append(f"`{command_prefix}approve always` to approve permanently")
    choices.append(f"`{command_prefix}deny` to cancel")
    return (
        f"{heading}\n```\n{cmd_preview}\n```\nReason: {description}\n\n"
        + ", ".join(choices[:-1]) + f", or {choices[-1]}.")

# Ordered: auth beats policy beats rate-limit beats connection; first match wins.
_PROVIDER_ERROR_REPLIES = (
    (_GATEWAY_AUTH_ERROR_RE, "⚠️ Provider authentication failed. Check the configured credentials; "
                             "raw provider details are in the gateway logs."),
    (_GATEWAY_PROVIDER_POLICY_RE, "⚠️ The model provider rejected the request. I kept the raw provider "
                                  "error out of chat; check gateway logs for details or try rephrasing."),
    (_GATEWAY_RATE_LIMIT_RE, "⏱️ The model provider is rate-limiting requests. Please wait a moment and try again."),
    (_GATEWAY_CONNECTION_ERROR_RE, "⚠️ The model server is not responding — it looks like the configured "
                                   "model endpoint is not running or is unreachable."))


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    for pattern, reply in _PROVIDER_ERROR_REPLIES:
        if pattern.search(text):
            return reply
    return (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics.")


# Provider/API failure envelope preambles (not ordinary assistant prose), anchored at line start.
_PROVIDER_ERROR_MARKERS = (
    r"api\s+(?:call\s+)?failed", r"provider\s+authentication\s+failed", r"non-retryable\s+error",
    r"rate\s+limited\s+after\s+\d+\s+retries", r"error\s+code\s*:", r"http\s*\d{3}\b",
    r"incorrect\s+api\s+key", r"invalid\s+api\s+key")
_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    + "|".join(_PROVIDER_ERROR_MARKERS + _CONNECTION_ERROR_MARKERS[:8] + (r"all\s+connection\s+attempts\s+failed",))
    + ")",
    re.IGNORECASE)


def _looks_like_gateway_provider_error(text: str) -> bool:
    """True when text is a provider failure envelope, not normal content.

    Must be short (envelopes are 1-3 lines) AND start with the marker, so prose citing a status code misses."""
    if not text:
        return False
    body = str(text).strip()
    if len(body) > 400 or body.count("\n") > 4:
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


def _sanitize_gateway_final_response(platform: Any, text: str) -> str:
    """Sanitize final gateway replies for chat surfaces: concise, secret-redacted provider failure
    categories instead of raw HTTP bodies, request IDs, leaked credentials, or policy text."""
    if not text or _gateway_surface_passes_raw_text(platform):
        return text

    # Lone UTF-16 surrogates make Telegram/Signal ``.encode()`` raise; last defense for legacy/plugin paths.
    # Lone UTF-16 surrogates (U+D800–U+DFFF) in model output crash chat surfaces downstream: Telegram's
    # ``utf16_len`` length check and Signal formatting both ``.encode()`` the reply and raise
    # UnicodeEncodeError before any send (#55143, #55309). The stored-history copy is already sanitized by
    # ``build_assistant_message`` and ``finalize_turn`` scrubs the returned ``final_response``, but this
    # boundary is the last line of defense for every legacy/plugin delivery path that hands us raw text.
    # Raw-text/programmatic surfaces above keep passthrough — their JSON consumers escape surrogates safely.
    from agent.message_sanitization import _sanitize_surrogates

    text = _sanitize_surrogates(str(text))

    # Cancellation metadata, not prose; ACP/TUI already suppress this sentinel, chat surfaces should too.
    # See #7921.
    if str(text).strip().startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX):
        return ""

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(redacted):
        return _gateway_provider_error_reply(redacted)
    return redacted


def _prepare_gateway_status_message(platform: Any, event_type: str, message: str) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery.

    Local/CLI keep the raw diagnostic stream; messaging surfaces drop transient aux/compression noise."""
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_surface_passes_raw_text(platform):
        return text

    text = _redact_gateway_user_facing_secrets(text)
    # Opt-in `compression.progress_notices` lets ROUTINE (template-derived) progress through; other noise stays.
    if _TELEGRAM_NOISY_STATUS_RE.search(text) and not (
        _gateway_compression_progress_notices_enabled() and _COMPRESSION_PROGRESS_STATUS_RE.search(text)
    ):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line (messaging has no status bar: one-shot push).

    The level glyph is already baked into the text (prepending would DOUBLE it); malformed/empty -> ""."""
    return str(getattr(notice, "text", "") or "").strip()


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status through adapter.send_or_update_status when supported (edits the previous
    bubble for the same status_key instead of appending); otherwise fall back to plain send.

    See #30045.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)


def _approval_send_outcome(future, timeout: float) -> str:
    """Classify an approval prompt send as ``sent`` / ``failed`` / ``ambiguous``.

    ``ambiguous`` = future timed out but the card may have posted: keep the registration, do NOT re-send.
    Only a DEFINITIVE failure (error result / non-timeout exception / no future) re-asks; logged here."""
    if future is None:
        logger.warning("Prompt send failed: no scheduling future (loop unavailable)")
        return "failed"
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return "ambiguous"
    except Exception as exc:
        logger.warning("Prompt send failed: %s", exc)
        return "failed"
    if getattr(result, "success", False):
        return "sent"
    # P5(b): a connector DECLINE is not a lane failure. The connector
    # authorized the destination and refused it; re-sending the same content as
    # plain text into that same chat is the exfiltration the egress guard
    # exists to stop. `failed` is the cue to fall back, so a decline needs its
    # own verdict — callers must surface it and send nothing further.
    #
    # CLASSIFY THE STRUCTURED RESPONSE, NOT THE ERROR STRING. The adapter
    # preserves the connector's own dict in `raw_response`; rebuilding a dict
    # from `error` alone loses two things review demonstrated:
    #   * a decline carrying `code: egress_declined` and NO text renders as
    #     "relay egress declined" — no marker colon — so the string check
    #     missed it and the fallback fired into the refused chat;
    #   * `ambiguous: True` (lost ack, mid-write drop) was flattened into a
    #     DEFINITE failure, which re-sends a card that may well have posted.
    # I fixed the text-marker path and tested only the text-marker path.
    from gateway.relay.egress import declined_send

    _raw = getattr(result, "raw_response", None)
    if isinstance(_raw, dict) and _raw.get("ambiguous"):
        # The frame may have been applied. Same physics as a scheduling
        # timeout: possibly-delivered, so never re-send. Checked BEFORE the
        # decline classification because an ambiguous result is a transport
        # outcome, not an authorization one, and this lane has three verdicts
        # rather than the boolean the shared helper answers.
        logger.warning("Prompt send AMBIGUOUS (lost ack): %s", _raw.get("error"))
        return "ambiguous"
    if declined_send(result):
        # Both shapes, one classifier: a structured body, or the uniform
        # decline sentence from an older connector.
        logger.warning(
            "Prompt send DECLINED by connector egress guard: %s",
            getattr(result, "error", None),
        )
        return "declined"
    logger.warning("Prompt send failed: %s", getattr(result, "error", None) or "unknown error")
    return "failed"


def _clarify_send_disposition(fut, *, session_key: str, clarify_mod) -> "str | None":
    """Decide whether a clarify prompt send aborts the wait; returns the abort sentinel or ``None``.

    Only a DEFINITIVE failure tears down the registration; ``ambiguous`` (card may have posted) stays armed
    and proceeds to the bounded wait, whose response timeout covers a lost card."""
    outcome = _approval_send_outcome(fut, timeout=15)
    if outcome == "declined":
        # P5(b): a connector DECLINE is MORE definitive than a failure — the
        # destination was authorized and refused, so the card cannot arrive and
        # no late reply can resolve it. Without this branch `declined` fell
        # through to the bounded wait and the agent blocked until
        # clarify_timeout (indefinitely when that is configured non-positive).
        logger.warning(
            "Clarify prompt DECLINED by the connector's egress guard; "
            "clearing registration"
        )
        clarify_mod.clear_session(session_key)
        return "[clarify prompt could not be delivered: destination refused]"
    if outcome == "failed":
        # Undeliverable: clear the registration and return the sentinel so the agent falls back, not hangs.
        logger.warning("Clarify send failed definitively; clearing registration")
        clarify_mod.clear_session(session_key)
        return "[clarify prompt could not be delivered]"
    if outcome == "ambiguous":
        logger.warning(
            "Clarify prompt send timed out — treating as possibly-delivered "
            "(no teardown; the registration stays armed for a late reply)")
    return None


def _clarify_send_then_wait(fut, *, clarify_id: str, session_key: str, clarify_mod) -> str:
    """Resolve a clarify prompt: send disposition, then the bounded wait."""
    abort = _clarify_send_disposition(fut, session_key=session_key, clarify_mod=clarify_mod)
    if abort is not None:
        return abort
    timeout = clarify_mod.get_clarify_timeout()
    response = clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
    if response is None or response == "":
        return f"[user did not respond within {int(timeout / 60)}m]"
    return response


def _resolve_progress_thread_id(
    platform: Any, source_thread_id: Any, event_message_id: Any, *, reply_in_thread: bool = True
) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target.

    ``reply_in_thread=False`` (Slack): no synthetic-thread fallback, else the final flat reply inherits a thread.
    A source.thread_id equal to the event's message id is the adapter's synthetic session key: no thread.

    See #18859.
    """
    platform_key = str(getattr(platform, "value", platform) or "").lower()
    if not reply_in_thread:
        if source_thread_id and event_message_id and str(source_thread_id) == str(event_message_id):
            return None
        return str(source_thread_id) if source_thread_id else None
    if source_thread_id:
        return str(source_thread_id)
    if platform_key in {"slack", "mattermost", "buzz"} and event_message_id:
        return str(event_message_id)
    return None


def _has_platform_display_override(user_config: dict, platform_key: str, setting: str) -> bool:
    """Return True when display.platforms.<platform> explicitly sets setting."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg


def _resolve_gateway_display_bool(
    user_config: dict, platform_key: str, setting: str, *, default: bool = False,
    platform: Any = None, require_platform_override_for: set[Any] | None = None) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Scratch-text is too noisy for threaded surfaces (Mattermost): they need an explicit per-platform override.
    """
    current_platform = _gateway_platform_value(platform or platform_key)
    platform_only = {_gateway_platform_value(c) for c in (require_platform_override_for or set())}
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)


def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid names (lowercase/digits/underscore); no-op elsewhere."""
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands_platforms import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)


# Auto-continue interrupted turns only while fresh, else stale tool-tail/resume_pending markers revive an old
# task after a restart. 1h covers agent.gateway_timeout (30 min) + slack; cfg agent.gateway_auto_continue_freshness.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60

# Boot auto-resume drain before the inbound gate opens. Override: agent.gateway_startup_restore_drain_timeout.
_STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT = 30.0

# Bound on the boot warm-up BEFORE the gate opens (no skeleton system prompt on turn one); keeps a wedged init
# from wedging the gateway. Override: ``agent.gateway_startup_warmup_timeout`` (non-positive disables).
_STARTUP_WARMUP_TIMEOUT_SECS_DEFAULT = 20.0


def _coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable -> None, so legacy transcripts keep auto-continuing instead of being dropped."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _auto_continue_freshness_window() -> float:
    """Auto-continue freshness window in seconds (non-positive disables the gate).

    Thin wrapper over ``gateway.session`` kept so ``gateway.run`` imports/test patches keep working."""
    from gateway.session_lifecycle import auto_continue_freshness_window
    return auto_continue_freshness_window()


def _startup_restore_drain_timeout_secs() -> float:
    """Max seconds ``_finish_startup_restore`` holds the inbound gate for boot auto-resume; <=0 disables.

    Duplicate-agent safety does NOT depend on it: ``_schedule_resume_pending_sessions`` claims SYNCHRONOUSLY.
    """
    return _float_env("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT)


def _startup_warmup_timeout_secs() -> float:
    """Max seconds the boot warm-up (``_warm_turn_prerequisites``) may hold the inbound gate shut.

    On timeout the gate opens and the warm-up finishes in the background. Non-positive disables it."""
    return _float_env("HERMES_STARTUP_WARMUP_TIMEOUT", _STARTUP_WARMUP_TIMEOUT_SECS_DEFAULT)


def _warm_turn_machinery_sync() -> int:
    """Synchronously initialize first-turn prerequisites (executor thread); returns the schema count.

    Covers the lazy init seen in skeleton turns: ``run_agent`` import graph, tool schemas (+ ``check_fn``
    TTL cache), context files, the local Python toolchain probe (#106064)."""
    import run_agent  # noqa: F401  # heavy import graph, cached in sys.modules
    import model_tools

    tool_defs = model_tools.get_tool_definitions(quiet_mode=True)
    try:
        from agent.prompt_builder import build_context_files_prompt

        build_context_files_prompt()
    except Exception:
        logger.debug("context-file warm-up failed (non-fatal)", exc_info=True)
    from hermes_cli.config import load_config_readonly

    agent_cfg = load_config_readonly().get("agent")
    if not isinstance(agent_cfg, dict) or agent_cfg.get("environment_probe", True):
        # The resolver owns remote-backend omission, the single worker, its cache and the bounded
        # wait; calling it here is what the first prompt build would otherwise do on the hot path.
        from tools.env_probe import get_environment_probe_line

        get_environment_probe_line()
    return len(tool_defs)


def _as_thread_info(info: Any) -> Optional[Tuple[str, str]]:
    """*info* as a (thread_id, initial_name) pair, or None if it isn't one.

    The pair crosses the relay connector boundary, so its shape is the connector's word, not ours."""
    if isinstance(info, tuple) and len(info) == 2 and all(isinstance(x, str) for x in info):
        return cast(Tuple[str, str], info)
    return None


def _float_env(name: str, default: float) -> float:
    """Read an env var as float; unset/empty/malformed fall back to ``default`` (never crash the gateway)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _stamp_hygiene_compression_provenance(
    agent: Any, desc: str, provenance: ActivityProvenance, debug_label: str) -> None:
    """Best-effort activity provenance stamp for hygiene compression transitions."""
    try:
        agent._touch_activity(desc, provenance=provenance)
    except Exception:
        logger.debug(debug_label, exc_info=True)


def _is_fresh_gateway_interruption(
    value: Any, *, now: Optional[float] = None, window_secs: Optional[float] = None) -> bool:
    """True when an interruption marker is fresh enough to auto-continue (unknown timestamps count as fresh)."""
    window = float(window_secs) if window_secs is not None else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    if window <= 0:
        return True
    timestamp = _coerce_gateway_timestamp(value)
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp <= window


def build_resume_recovery_note(
    reason: Optional[str], message: str = "", *, interactive: bool = True) -> str:
    """Build the resume-pending recovery system note for an interrupted turn (empty ``message`` = auto-resume).

    Interactive platforms report the restore and ask what next; non-interactive ones finish the work.

    On non-interactive event platforms (webhook, API server — adapters with ``interactive_resume = False``)
    nobody can answer; the resumed turn must instead complete the interrupted work, or the task is silently
    abandoned behind a "restored" acknowledgement that goes nowhere (#57056).
    """
    reason_phrase = (
        "a gateway restart" if reason == "restart_timeout"
        else "a gateway shutdown" if reason == "shutdown_timeout" else "a gateway interruption")
    if message:
        resume_guidance = (
            "Address the user's NEW message below FIRST and focus on what the user is asking now.")
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any unfinished work from the conversation history."
        )
    elif interactive:
        resume_guidance = (
            "Report to the user that the session was restored "
            "successfully and ask what they would like to do next.")
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any unfinished work from the conversation history."
        )
    else:
        resume_guidance = (
            "No user is present on this non-interactive platform, "
            "so do NOT emit a 'session restored' acknowledgement "
            "or ask questions. Review the conversation history and "
            "CONTINUE the interrupted task to completion.")
        tail_guidance = (
            "Do NOT re-run tool calls whose results already "
            "appear in the history — resume from the first step that has no recorded result.")
    return (
        f"[System note: The previous turn was interrupted by "
        f"{reason_phrase}; the gateway is now back online. "
        f"Any restart/shutdown command in the history has already "
        f"run — do NOT re-execute or verify it. {resume_guidance} {tail_guidance}]"
        + (f"\n\n{message}" if message else ""))


def _prepare_resume_pending_message(
    reason: Optional[str], message: Optional[str], *, interactive: bool = True) -> tuple[str, str]:
    """Return the recovery message and the user text to persist.

    Empty original: persist the note (a "" user row trips the pre-call sanitizer). Real text: persist clean.

    Resume turns replace the startup event's text with a recovery note before entering the agent. When the
    original message is empty (the synthesized auto-resume turn), persist the note too — persisting the
    empty string left a blank user row in state.db that the pre-call sanitizer re-healed on every later call
    forever (#86580). When the user sent REAL text while the resume was pending, keep persisting their clean
    words: the transcript stays scaffold-free (the model still receives the wrapped note), and a non-empty
    row never trips the sanitizer.
    """
    recovery_message = build_resume_recovery_note(reason, message or "", interactive=interactive)
    persist_message = message if isinstance(message, str) and message.strip() else recovery_message
    return recovery_message, persist_message


# Assistant fields that must survive replay for CLI parity (reasoning continuity, prefix-cache hits, provider
# echo): unreconstructable thinking text (DeepSeek/Kimi), opaque signatures, Codex blobs (caching degrades).
# ``reasoning`` and ``reasoning_details`` were the original three preserved by PR #2974 (schema v6).
# ``reasoning_content``, ``codex_reasoning_items``, ``codex_message_items``, and ``finish_reason`` were
# added to the DB later but the gateway's replay whitelist was never expanded to match — so any pure-text
# assistant turn (no ``tool_calls``) silently dropped them on replay, regressing the CLI-vs-gateway
# behavioural parity. Why each field matters on replay: ``_copy_reasoning_content_for_api`` promotes
# ``reasoning`` → ``reasoning_content`` at send time, but only when the strings happen to match. Carrying
# the original ``reasoning_content`` verbatim avoids reconstruction loss for providers that return them as
# distinct fields (DeepSeek/Kimi/Moonshot thinking modes). * ``reasoning_details``: opaque structured array
# (signature, encrypted_content) used by OpenRouter/Anthropic to maintain reasoning continuity across turns.
# * ``codex_reasoning_items``: encrypted reasoning blobs for the OpenAI Codex Responses API. *
# ``codex_message_items``: exact assistant message items with ``phase``. OpenAI docs: "preserve and resend
# phase on all assistant messages — dropping it can degrade performance."  Required for prefix cache hits. *
# ``finish_reason``: informational; cheap to keep so transcripts replay identically across CLI and gateway.
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "finish_reason")


def _build_replay_entry(
    role: str, content: Any, msg: Dict[str, Any], preserve_timestamp: bool = False
) -> Dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving ``_ASSISTANT_REPLAY_FIELDS``.

    ``preserve_timestamp``: only user rows need it (stale-dangerous-confirmation stripper). Falsy fields are
    dropped EXCEPT ``reasoning_content``: DeepSeek/Kimi treat "" as a sentinel; dropping it can 400.

    Empty values: most fields are dropped when falsy (matching the original PR #2974 behaviour) since an
    empty list/string for those carries no information. The exception is ``reasoning_content``:
    DeepSeek/Kimi thinking-mode replay treats an empty string as a meaningful sentinel that
    ``_copy_reasoning_content_for_api`` upgrades to a single space. Dropping it here would make the gateway
    send no ``reasoning_content`` at all on the next turn, which can cause HTTP 400 from strict thinking
    providers.
    """
    entry: Dict[str, Any] = {"role": role, "content": content}
    # api_content sidecar keeps the request prefix byte-stable — ONLY if this pipeline did not rewrite
    # content. The caller renders timestamps AFTER this check so a stamp alone never drops the sidecar.
    _sidecar = msg.get("api_content")
    if (
        role in ("user", "assistant")
        and isinstance(_sidecar, str)
        and _sidecar
        and content == msg.get("content")):
        entry["api_content"] = _sidecar
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if (_rval is None) if _rkey == "reasoning_content" else (not _rval):
                continue
            entry[_rkey] = _rval
    if preserve_timestamp and msg.get("timestamp"):
        entry["timestamp"] = msg["timestamp"]
    return entry


_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
_OBSERVED_GROUP_CONTEXT_HEADER = "[Observed Telegram group context - context only, not requests]"
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def _uses_telegram_observed_group_context(channel_prompt: Optional[str]) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Observed rows must not replay as ordinary user turns, or a weak wake word makes old chatter look like work.
    """
    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)


def _csv_or_list_to_set(raw: Any) -> set[str]:
    """Normalize a config list or comma-separated scalar into a string set."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _slack_ignored_channels_from_gateway_config(config: Any) -> set[str]:
    """Return Slack channels that the generic gateway must never dispatch.

    Duplicates the adapter's drop as a fail-safe so bypasses can't reach auth, pairing or sessions."""
    platform_cfg = getattr(config, "platforms", {}).get(Platform.SLACK)
    raw = None
    if platform_cfg is not None:
        raw = getattr(platform_cfg, "extra", {}).get("ignored_channels")
    if raw is None:
        # Top-level ``slack.ignored_channels`` arrives via the plugin's YAML→env bridge, not PlatformConfig.extra.
        # See #46925.
        raw = os.getenv("SLACK_IGNORED_CHANNELS") or None
    return _csv_or_list_to_set(raw)


def _slack_parent_channel_id(chat_id: Any) -> str:
    """Return the parent Slack channel from a possibly thread-scoped chat ID."""
    return str(chat_id).split(":", 1)[0] if chat_id else ""


def _is_slack_ignored_channel(config: Any, chat_id: Any) -> bool:
    """Check the generic Slack gateway blacklist for channel or thread IDs."""
    channel_id = _slack_parent_channel_id(chat_id)
    ignored = _slack_ignored_channels_from_gateway_config(config)
    return bool(channel_id and ("*" in ignored or channel_id in ignored))


def _message_timestamps_enabled(user_config: Optional[dict]) -> bool:
    """True when gateway.message_timestamps.enabled is opted in (default OFF: changes what the model sees)."""
    if not isinstance(user_config, dict):
        return False
    gw = user_config.get("gateway")
    if not isinstance(gw, dict):
        return False
    mt = gw.get("message_timestamps")
    if isinstance(mt, dict):
        return bool(mt.get("enabled", False))
    # Allow a bare ``message_timestamps: true`` shorthand.
    return bool(mt)


def _build_gateway_agent_history(
    history: List[Dict[str, Any]], *, channel_prompt: Optional[str] = None,
    inject_timestamps: bool = False) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Convert stored gateway transcript rows into agent replay messages.

    Observed context stays out of ``conversation_history`` so consecutive-user repair can't merge it in."""
    from hermes_time import get_timezone as _get_msg_tz
    from gateway.message_timestamps import (
        render_user_content_with_timestamp as _render_msg_ts,
        strip_leading_message_timestamps as _strip_msg_ts,
    )

    _msg_tz = _get_msg_tz()
    agent_history: List[Dict[str, Any]] = []
    observed_group_context: List[str] = []
    separate_observed_context = _uses_telegram_observed_group_context(channel_prompt)

    for msg in history or []:
        role = msg.get("role")
        # session_meta rows are transcript logging, not LLM input; the agent rebuilds its own system prompt.
        if not role or role in {"session_meta", "system"}:
            continue

        content = msg.get("content")
        if separate_observed_context and msg.get("observed") and role == "user" and content:
            if inject_timestamps and isinstance(content, str):
                content = _render_msg_ts(content, msg.get("timestamp"), tz=_msg_tz)
            observed_group_context.append(str(content).strip())
            continue

        # Rich tool_calls/tool-result rows pass through intact so the API sees valid assistant→tool sequences.
        if "tool_calls" in msg or "tool_call_id" in msg or role == "tool":
            clean_msg = {k: v for k, v in msg.items() if k not in {"timestamp", "observed"}}
            agent_history.append(clean_msg)
        elif content:
            replay_timestamp = msg.get("timestamp")
            # Clean before rendering: a timestamp prefix hides recovery notes
            # from the startswith-based stripper. Retain an embedded original time.
            if role == "user":
                if isinstance(content, str):
                    body, embedded_timestamp = _strip_msg_ts(content, tz=_msg_tz)
                    clean_body = _strip_auto_continue_noise(body)
                    if clean_body != body:
                        content = clean_body
                        if embedded_timestamp is not None:
                            replay_timestamp = embedded_timestamp
                if not content:
                    continue
            # Keep user timestamps for the stale-dangerous-confirmation stripper in agent/replay_cleanup.py.
            entry = _build_replay_entry(role, content, msg, preserve_timestamp=(role == "user"))
            if inject_timestamps and role == "user" and isinstance(content, str):
                rendered = _render_msg_ts(content, replay_timestamp, tz=_msg_tz)
                # Preserve only a sidecar matching the complete rendered message,
                # optionally followed by the normal context separator. Cleanup
                # above already invalidated sidecars containing stripped content.
                sidecar = entry.get("api_content")
                if rendered != content and sidecar and not (
                    sidecar == rendered or sidecar.startswith(rendered + "\n\n")
                ):
                    entry.pop("api_content", None)
                entry["content"] = rendered
            if msg.get("mirror"):
                mirror_src = msg.get("mirror_source", "another session")
                entry["content"] = f"[Delivered from {mirror_src}] {entry['content']}"
                entry.pop("api_content", None)  # prefix rewrite: the sidecar no longer matches
            agent_history.append(entry)

    # Strip interrupted tool-call tails so the LLM doesn't re-execute tools killed mid-flight.
    agent_history = strip_interrupted_tool_tails(agent_history)

    # Strip a dangling assistant(tool_calls) tail (SIGKILL-mid-tool-call); else the model re-issues it forever.
    # Strip a dangling assistant(tool_calls) tail with no tool answers — the signature of a SIGKILL
    # mid-tool-call (e.g. the tool itself ran `docker restart`/`kill` and took the gateway down before the
    # result was persisted). Without this the model re-issues the unanswered call on resume and loops the
    # restart forever (#49201).
    agent_history = strip_dangling_tool_call_tail(agent_history)

    # Strip expired dangerous-confirmation phrases; replayed, a follow-up could read as a fresh confirmation.
    agent_history = strip_stale_dangerous_confirmations(agent_history, now=time.time())

    observed_context = "\n".join(observed_group_context).strip() or None
    return agent_history, observed_context


def _select_cached_agent_history(
    persisted_history: List[Dict[str, Any]], live_history: Any) -> List[Dict[str, Any]]:
    """Prefer the cached live transcript only when it is longer AND has a real, non-ephemeral unpersisted row.

    Guards FTS write-corruption amnesia (stale reload while the cached agent holds unpersisted rows). Length
    alone is not enough: a longer all-durable list can be an expected replay-filtering delta.

    Guards the FTS write-corruption case (#50502): when message writes fail silently through corrupt FTS
    triggers, the next turn reloads a stale/empty ``conversation_history`` from disk even though the same
    cached ``AIAgent`` still holds unpersisted real rows in ``_session_messages``. Replacing those rows with
    the shorter persisted copy causes immediate same-session amnesia. Length alone does not trigger
    retention.
    """
    if isinstance(live_history, list) and len(live_history) > len(persisted_history):
        from agent.session_persistence import _is_ephemeral_scaffolding

        has_unpersisted_row = any(
            isinstance(message, dict) and not message.get("_db_persisted")
            and not _is_ephemeral_scaffolding(message) for message in live_history)
        if has_unpersisted_row:
            return list(live_history)
    return persisted_history


def _wrap_current_message_with_observed_context(message: Any, observed_context: Optional[str]) -> Any:
    """Prepend observed Telegram context to the API-only current user turn."""
    if not observed_context:
        return message

    prefix = f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n{observed_context}\n\n{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"

    if isinstance(message, str):
        return f"{prefix}{message}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}"
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped

    return message


def _last_transcript_timestamp(history: Optional[List[Dict[str, Any]]]) -> Any:
    """Return the ``timestamp`` of the last usable (non-metadata) transcript row, if any.

    ``None`` when the last usable row has no timestamp — callers treat that as "fresh" (legacy rows)."""
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        ts = msg.get("timestamp")
        if ts is not None:
            return ts
        return None
    return None


# Tool output may hold literal MEDIA: examples (docs, logs); only deliberate media producers may auto-append.
_AUTO_APPEND_MEDIA_TOOL_NAMES = {"text_to_speech", "text_to_speech_tool", "image_generate"}

# Replay-tail sanitization lives in agent/replay_cleanup.py so every resume surface shares one implementation.
from agent.replay_cleanup import (  # noqa: E402
    strip_interrupted_tool_tails, strip_dangling_tool_call_tail, strip_stale_dangerous_confirmations)


_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn"
_AUTO_CONTINUE_FALLBACK_PREFIX = "[System note: A new message"


def _is_auto_continue_noise(content: Any) -> bool:
    """Return True if this user-message content is a gateway-injected auto-continue note (never replay it)."""
    return isinstance(content, str) and content.startswith(
        (_AUTO_CONTINUE_NOTE_PREFIX, _AUTO_CONTINUE_FALLBACK_PREFIX))


def _strip_auto_continue_noise(content: Any) -> Any:
    """Strip leading persisted auto-continue notes from user text; the trailing real question is preserved."""
    if not _is_auto_continue_noise(content):
        return content
    text = str(content)
    while _is_auto_continue_noise(text):
        end = text.find("]")
        if end < 0:
            return ""
        text = text[end + 1 :].lstrip()
    return text

# Tools whose deliverable is a JSON payload with a local-file path field rather than a literal ``MEDIA:`` tag.
_JSON_MEDIA_TOOL_PATH_FIELDS = ("host_image", "image", "agent_visible_image")


# Extension-anchored MEDIA: matcher (mirrors the dispatch site); a bare ``MEDIA:`` in prose never auto-appends.
_TOOL_MEDIA_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE)


# Shared with cron delivery and gateway background tasks; canonical names live in gateway.media_repair.
from gateway.media_repair import tool_name_by_call_id as _tool_name_by_call_id  # noqa: E402


def _collect_auto_append_media_tags(
    messages: List[Dict[str, Any]], history_offset: int = 0,
    history_media_paths: Optional[set] = None) -> tuple[List[str], bool]:
    """Collect real media tags from current-turn producer-tool results only.

    Producer allowlist: docs/logs/search results contain example MEDIA: strings that must never become
    attachments. If mid-run compression shrank the list below the history length the slice is
    untrustworthy, so scan every message (dedup via history_media_paths).

    1. Producer-tool allowlist: only tools that intentionally emit deliverable artifacts (TTS) are eligible.
    (Fixes the original report behind #16721.) 2. Current-turn isolation: only messages produced this turn
    are scanned, so a tool result from an earlier turn (still present in the full message list) cannot leak
    onto a later text-only reply (#34608).
    When that happens the slice boundary is no longer trustworthy, so fall back to scanning every message
    and rely on ``history_media_paths`` for dedup, preserving the compression-safe behaviour of #160. The
    producer-tool allowlist still applies on the fallback path.
    """
    history_media_paths = history_media_paths or set()
    new_messages = (messages[history_offset:]
                    if history_offset and len(messages) >= history_offset else messages)

    tool_name_by_call_id = _tool_name_by_call_id(new_messages)

    media_tags: List[str] = []
    has_voice_directive = False
    for msg in new_messages:
        if msg.get("role") not in ("tool", "function"):
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(call_id) not in _AUTO_APPEND_MEDIA_TOOL_NAMES:
            continue
        content = str(msg.get("content") or "")
        tool_name = tool_name_by_call_id.get(call_id)
        # image_generate emits a JSON path field, not a MEDIA: tag; extract it: deterministic delivery.
        if tool_name == "image_generate" and "MEDIA:" not in content:
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    path = payload.get(field)
                    if (isinstance(path, str)
                            and _TOOL_MEDIA_RE.fullmatch(f"MEDIA:{path}")
                            and path not in history_media_paths):
                        media_tags.append(f"MEDIA:{path}")
                        break
            continue
        if "MEDIA:" not in content:
            continue
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path and path not in history_media_paths:
                media_tags.append(f"MEDIA:{path}")
        if "[[audio_as_voice]]" in content:
            has_voice_directive = True

    return media_tags, has_voice_directive


def _collect_history_media_paths(agent_history: List[Dict[str, Any]]) -> set:
    """Dedup set of media paths already delivered (JSON-payload and assistant-message shapes alike).

    Missing the JSON-payload shape caused #46627; missing the assistant-message shape caused repeated
    delivery when the model echoed a previous MEDIA tag.
    """
    paths: set = set()
    tool_name_by_call_id = _tool_name_by_call_id(agent_history)

    def _add_text_media_paths(content: str) -> None:
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path:
                paths.add(path)
        # The regex misses quoted/spaced paths extract_media accepts; use the same extractor to dedup.
        media_files, _ = BasePlatformAdapter.extract_media(content)
        paths.update(path for path, _is_voice in media_files)

    for msg in agent_history:
        role = msg.get("role")
        if role not in ("assistant", "tool", "function"):
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" in content:
            _add_text_media_paths(content)
            continue
        if role == "assistant":
            continue
        cid = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(cid) == "image_generate":
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    jp = payload.get(field)
                    if isinstance(jp, str) and jp:
                        paths.add(jp)
                        break
    return paths

def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE when the system hides CA certs from Python (NixOS etc.); must run BEFORE any
    HTTP library is imported. A set-but-missing path breaks every later httpx client: treat as unset."""
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist", configured_cert)
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

def _home_target_env_var(platform_name: str) -> str:
    """Home-target env var: built-in ``_HOME_TARGET_ENV_VARS``, plugin registry, then
    ``<PLATFORM>_HOME_CHANNEL``."""
    from cron.scheduler_delivery import _resolve_home_env_var
    return _resolve_home_env_var(platform_name) or f"{platform_name.upper()}_HOME_CHANNEL"


def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"


def _restart_notification_pending() -> bool:
    """Return True when a /restart completion marker is waiting to be delivered."""
    return (_hermes_home / ".restart_notify.json").exists()


def _planned_restart_notification_path() -> Path:
    return _hermes_home / ".restart_pending.json"


def _planned_restart_notification_pending() -> bool:
    """Return True when a non-chat planned restart should notify home channels."""
    return _planned_restart_notification_path().exists()


def _clear_planned_restart_notification() -> None:
    _planned_restart_notification_path().unlink(missing_ok=True)


# Gateway marker so a lazily imported cli.py load_cli_config() doesn't clobber TERMINAL_CWD.
os.environ["_HERMES_GATEWAY"] = "1"

_ensure_ssl_certs()

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home, get_hermes_home_override
_hermes_home = get_hermes_home()

# Load ~/.hermes/.env first: user-managed env files must override stale shell exports on restart.
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')


def _reload_runtime_env_preserving_config_authority() -> None:
    """Reload .env per turn for rotated keys while config.yaml stays authoritative for budgets (else a
    stale HERMES_MAX_ITERATIONS wins). Multiplex never reloads .env globally: secrets come from the
    per-turn ``set_secret_scope`` and mutating ``os.environ`` would leak the default profile's keys to
    every profile; it still honors the max_turns bridge."""
    from agent.secret_scope import is_multiplex_active
    if not is_multiplex_active():
        load_hermes_dotenv(
            hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')
    _bridge_max_turns_from_config(_hermes_home)


def _bridge_max_turns_from_config(home: "Path") -> None:
    """Re-bridge agent.max_turns (+ sessions.*) per turn; managed overlay applies or it reverts."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        cfg = _load_bridge_config(config_path)
    except Exception:
        return
    _bridge_max_turns_to_env(cfg.get("agent", {}))
    _bridge_section_to_env(cfg.get("sessions", {}), _SESSIONS_ENV_BRIDGE)


def _current_max_iterations() -> int:
    """Return the per-turn iteration budget after runtime env refresh; ``resolve_turn_limit`` maps
    ``agent.max_turns: none``/``unlimited`` (bridged as a string) to the unlimited sentinel, not an
    ``int()`` crash."""
    _reload_runtime_env_preserving_config_authority()
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    return _resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))


from contextlib import asynccontextmanager as _asynccontextmanager, contextmanager as _contextmanager, suppress


class MultiplexConfigError(RuntimeError):
    """Invalid profile multiplexer config: the operator must fix config.yaml, so it propagates to the
    startup guard instead of being treated as retryable adapter-connect noise."""


class SecondaryPortBindingConfigError(MultiplexConfigError):
    """A secondary profile enabled a port-binding platform: the default profile owns the single shared
    listener (/p/<profile>/), so this is always a misconfiguration and is skipped, not fatal."""


class HygieneTurnHoldExceeded(Exception):
    """Hygiene-compression turn-hold budget elapsed mid-stream. Availability boundary, not a failure:
    must NOT take the idle-timeout path (AGENT_COMPRESSION_TIMEOUT, "no output", failure cooldown)."""


def _multiplex_profile_homes(config: object) -> list[tuple[str, "Path"]]:
    """Return the authoritative profile set for one multiplex gateway config."""
    from hermes_cli.profiles import profiles_to_serve
    return list(profiles_to_serve(
        multiplex=True, profile_allowlist=getattr(config, "multiplex_profile_allowlist", None)))


def _enable_multiplex_log_routing(config: object) -> bool:
    """Route agent.log/errors.log/gateway.log records to their owning profile (inert single-profile).
    ``setup_logging(mode="gateway")`` binds file handlers to the launch home, so under multiplexing
    every secondary profile's records would land in the default profile's logs.

    Swap the static handlers for the profile routers from #99440 — the same primitive the Desktop cron
    ticker uses — once the served-profile set is known. Inert for single-profile gateways
    (``enable_profile_log_routing`` is a no-op below two homes).
    """
    if not getattr(config, "multiplex_profiles", False):
        return False
    try:
        from hermes_logging import enable_profile_log_routing
        return enable_profile_log_routing([home for _name, home in _multiplex_profile_homes(config)])
    except Exception:
        logger.debug("could not enable per-profile log routing", exc_info=True)
        return False


def _handoff_watch_scopes(runner: object) -> list:
    """``(profile_name, home)`` pairs whose ``state.db`` the watcher must poll; ``(None, None)`` = root
    poll, always first. ``/handoff`` writes into the store of the profile the CLI ran under; an unscoped
    watcher polls only the ROOT store, so a secondary profile's handoff would never be seen (CLI times
    out). A raising resolver degrades to the root poll rather than silently disabling the watcher."""
    scopes: list = [(None, None)]
    try:
        config = getattr(runner, "config", None)
        if config is not None and getattr(config, "multiplex_profiles", False):
            for name, home in _multiplex_profile_homes(config):
                if home is None or not name or name == "default":
                    continue
                scopes.append((name, home))
    except Exception:
        logger.debug("Could not resolve multiplex homes for handoff watcher", exc_info=True)
    return scopes


async def _reclaim_stale(runner: object) -> None:
    """Fail handoffs left in ``running`` by a gateway that died mid-dispatch (once per store at startup).
    ``running`` is only set for one in-process dispatch, so a leftover row belongs to a dead process and
    blocks ``request_handoff`` for that session forever. Defensive: a raising reclaim aborts startup."""
    reclaim = getattr(getattr(runner, "_session_db", None), "reclaim_stale_running_handoffs", None)
    if not callable(reclaim):
        return
    try:
        ids = await reclaim(
            "gateway stopped mid-handoff; state reclaimed at startup. Re-run /handoff to try again.")
    except Exception:
        logger.debug("Stale-handoff reclaim raised", exc_info=True)
        return
    if ids:
        logger.warning(
            "Reclaimed %d handoff(s) stranded in 'running' by a previous "
            "gateway: %s", len(ids), ", ".join(str(i) for i in ids))


def _terminal_scope_cwd(default: str = "") -> str:
    """Scope-aware TERMINAL_CWD read for footer/context surfaces. Only an import failure falls back:
    an active refusal scope must raise, not use the launch cwd."""
    try:
        from tools.terminal_scope import terminal_env as _ts_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", default)
    return _ts_env("TERMINAL_CWD", default)


def _load_profile_secret_scope(profile_home: "Path") -> dict:
    """Hydrate and load one profile's secrets under its home override."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    # Caller already hydrated external sources off-loop (#99519).
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(str(profile_home))
    try:
        hydrate_profile_secret_sources(Path(profile_home))
        return build_profile_secret_scope(Path(profile_home))
    finally:
        reset_hermes_home_override(home_token)


@_contextmanager
def _profile_runtime_scope(
    profile_home: "Path", prepared_secret_scope: Optional[dict] = None, *,
    hydrate_secrets: bool = True):
    """Scope config/skills/memory AND credentials to a profile for one turn (multiplexed path only).
    ``set_hermes_home_override`` is a contextvar (reaches the agent worker via ``copy_context()``);
    ``set_secret_scope`` makes the profile ``.env`` the credential source without mutating
    ``os.environ``, so subprocesses never inherit cross-profile secrets."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope

    home_token = set_hermes_home_override(str(profile_home))
    if prepared_secret_scope is not None:
        secrets = prepared_secret_scope
    elif hydrate_secrets:
        secrets = _load_profile_secret_scope(Path(profile_home))
    else:
        from agent.secret_scope import build_profile_secret_scope  # caller already hydrated off-loop
        secrets = build_profile_secret_scope(Path(profile_home))
    secret_token = set_secret_scope(secrets)
    # Install the routed profile's COMPLETE terminal policy, never ambient TERMINAL_* a prior turn set.
    # Without it terminal_tool reads the process-global TERMINAL_* vars a previous profile's turn may have
    # pinned (first-writer-wins backend leak; #68559).
    from tools.terminal_scope import install_and_reset_profile_terminal_scope

    with install_and_reset_profile_terminal_scope(Path(profile_home)):
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)


@_asynccontextmanager
async def _async_profile_runtime_scope(profile_home: "Path"):
    """Enter a profile scope without loading secret files on the event loop."""
    secrets = await asyncio.to_thread(_load_profile_secret_scope, Path(profile_home))
    with _profile_runtime_scope(Path(profile_home), secrets):
        yield


def load_gateway_config_for_runner() -> "GatewayConfig":
    """Load gateway config for the process-level GatewayRunner. Multiplexed: reload under the default
    profile's ``_profile_runtime_scope`` so platform tokens in its ``.env`` resolve via the secret
    scope; unscoped ``_getenv`` falls to ``os.environ``, which often lacks a token living only under
    ``profiles/<name>/.env``. Off -> identical to ``load_gateway_config()``.

    See #64674.
    """
    cfg = load_gateway_config()
    if not getattr(cfg, "multiplex_profiles", False):
        return cfg
    try:
        home = get_hermes_home()
    except Exception:
        return cfg
    try:
        with _profile_runtime_scope(Path(home)):
            return load_gateway_config()
    except Exception:
        logger.debug("multiplex default-scope config reload failed; using unscoped load", exc_info=True)
        return cfg


async def _discover_gateway_mcp_tools(config: object) -> None:
    """Run startup MCP discovery for every profile this gateway serves: ``discover_mcp_tools`` reads
    ``mcp_servers`` from ``get_hermes_home()``'s config, so an unscoped call only connects the launch
    profile's servers (single-profile gateways keep the unscoped call).

    Under multiplex, run it once per served profile inside that profile's ``_profile_runtime_scope`` and
    carry the scope into the executor thread with ``copy_context()`` (the same shape as
    ``_run_in_executor_with_context``). See #95518.
    """
    from tools.mcp_tool_discovery import discover_mcp_tools
    loop = asyncio.get_running_loop()
    if not getattr(config, "multiplex_profiles", False):
        await loop.run_in_executor(None, discover_mcp_tools)
        return
    for profile_name, profile_home in _multiplex_profile_homes(config):
        try:
            with _profile_runtime_scope(Path(profile_home)):
                await loop.run_in_executor(None, copy_context().run, discover_mcp_tools)
        except Exception:
            logger.warning("MCP tool discovery failed for profile '%s'", profile_name, exc_info=True)


def _platform_has_bot_credential(platform: "Platform", platform_config: "PlatformConfig") -> bool:
    """Return True when a token-authenticated platform has a usable bot credential; platforms not using
    ``PlatformConfig.token`` (Signal session paths, port-binding HTTP adapters) always return True."""
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES, Platform
    if platform not in PLATFORM_TOKEN_ENV_NAMES:
        return True
    for attr in ("token", "api_key"):  # some adapters accept api_key as the primary credential
        value = getattr(platform_config, attr, None) or ""
        if isinstance(value, str) and value.strip():
            return True
    # Matrix also authenticates by password; a token-only check would evict a reconnectable config from
    # the retry queue. Read ONLY extra (build_config() copies env there): env fallback = every config OK.
    # Those credentials land in ``extra`` rather than ``.token``, so a token-only check reads a perfectly
    # reconnectable password-auth config as credential-less and evicts it from the retry queue on the first
    # transient failure — after which it stays down until the gateway is restarted by hand. Mirror the
    # adapter's own gate: homeserver + user_id + password. Read ONLY from extra, never os.getenv:
    # build_config() already copies all three env vars onto extra, and importing this module loads
    # ~/.hermes/.env, so an env fallback would report "has credential" for every Matrix config on the box —
    # including the empty-primary multiplex case (#64674) this check exists to evict.
    if platform is not Platform.MATRIX:
        return False
    extra = getattr(platform_config, "extra", None) or {}
    return all(str(extra.get(key) or "").strip() for key in ("homeserver", "user_id", "password"))


_DOCKER_VOLUME_SPEC_RE = re.compile(r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$")
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}

# Internal bridge, not a config source: seed from the canonical default after dotenv so an ambient
# process/.env value can never control lease safety.
from hermes_cli.config_defaults import DEFAULT_CONFIG as _DEFAULT_CONFIG
os.environ["HERMES_TURN_LEASE_TIMEOUT"] = str(_DEFAULT_CONFIG["agent"]["gateway_turn_lease_timeout"])

# Bridge config.yaml values into env so os.getenv() picks them up. config.yaml unconditionally wins
# over .env for these keys; a `not in os.environ` guard would let stale .env entries shadow config.
_AGENT_ENV_BRIDGE = {
    "gateway_timeout": "HERMES_AGENT_TIMEOUT",
    "gateway_turn_lease_timeout": "HERMES_TURN_LEASE_TIMEOUT",
    "gateway_timeout_warning": "HERMES_AGENT_TIMEOUT_WARNING",
    "gateway_notify_interval": "HERMES_AGENT_NOTIFY_INTERVAL",
    "session_stall_timeout": "HERMES_SESSION_STALL_TIMEOUT",
    # Internal bridge only — config.yaml (agent.reconnect_attention_after) is the documented setting.
    "reconnect_attention_after": "HERMES_RECONNECT_ATTENTION_AFTER_SECONDS",
    "restart_drain_timeout": "HERMES_RESTART_DRAIN_TIMEOUT",
    "cron_drain_timeout": "HERMES_CRON_DRAIN_TIMEOUT",
    "gateway_auto_continue_freshness": "HERMES_AUTO_CONTINUE_FRESHNESS",
    "gateway_startup_restore_drain_timeout": "HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT",
    "gateway_startup_warmup_timeout": "HERMES_STARTUP_WARMUP_TIMEOUT"}
# config-authoritative knobs for the session-search index (env stays the cross-process carrier).
_SESSIONS_ENV_BRIDGE = {"cjk_fts": "HERMES_CJK_FTS", "search_slow_ms": "HERMES_SEARCH_SLOW_MS"}
_DISPLAY_ENV_BRIDGE = {
    "busy_input_mode": "HERMES_GATEWAY_BUSY_INPUT_MODE",
    "busy_text_mode": "HERMES_GATEWAY_BUSY_TEXT_MODE",
    "busy_ack_enabled": "HERMES_GATEWAY_BUSY_ACK_ENABLED"}


def _bridge_section_to_env(section: Any, mapping: Dict[str, str]) -> None:
    """Export every present ``mapping`` key of a config section as ``str(value)``."""
    if isinstance(section, dict):
        for cfg_key, env_var in mapping.items():
            if cfg_key in section:
                os.environ[env_var] = str(section[cfg_key])


def _bridge_max_turns_to_env(agent_cfg: Any) -> None:
    """Bridge ``agent.max_turns`` preserving its raw spelling ("none", "unlimited", "120"); Python None
    (`null` / bare `key:`) clears a stale bridge instead, since str(None) -> "None" would map to the
    unlimited sentinel rather than "absent = default"."""
    if not isinstance(agent_cfg, dict) or "max_turns" not in agent_cfg:
        return
    raw = agent_cfg["max_turns"]
    if raw is not None:
        os.environ["HERMES_MAX_ITERATIONS"] = str(raw)
    elif "HERMES_MAX_ITERATIONS" in os.environ:
        del os.environ["HERMES_MAX_ITERATIONS"]


def _bridge_terminal_config_to_env(_terminal_cfg: dict) -> None:
    """Bridge nested ``terminal.*`` config to TERMINAL_* env vars (config.yaml overrides .env here)."""
    _terminal_backend = str(
        _terminal_cfg.get("backend") or os.environ.get("TERMINAL_ENV") or "").strip().lower()
    _terminal_env_map = {
        "backend": "TERMINAL_ENV",
        "degraded_mode": "TERMINAL_DEGRADED_MODE",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "home_mode": "TERMINAL_HOME_MODE",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
        "docker_env": "TERMINAL_DOCKER_ENV",
        "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
        "docker_shm_size": "TERMINAL_DOCKER_SHM_SIZE",
        "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "docker_network": "TERMINAL_DOCKER_NETWORK",
        "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
        "docker_snap_compat": "TERMINAL_DOCKER_SNAP_COMPAT",
        "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
        "docker_shared_container_key": "TERMINAL_DOCKER_SHARED_CONTAINER_KEY",
        "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL"}
    for _cfg_key, _env_var in _terminal_env_map.items():
        if _cfg_key not in _terminal_cfg:
            continue
        _val = _terminal_cfg[_cfg_key]
        if _cfg_key == "cwd":
            # Placeholders (".", "auto", "cwd") resolve to Path.home() later; only explicit paths bridge.
            if str(_val) in {".", "auto", "cwd"}:
                continue
            # Expand "~" for local/container cwd so Popen never gets a literal "~/" (kernel rejects it);
            # SSH cwd is interpreted by the remote shell: keep "~". Predicate shared w/ terminal_tool.
            if isinstance(_val, str) and not _is_ssh_remote_tilde_cwd(_terminal_backend, _val.strip()):
                _val = os.path.expanduser(_val)
        os.environ[_env_var] = json.dumps(_val) if isinstance(_val, (list, dict)) else str(_val)


def _bridge_auxiliary_config_to_env(_auxiliary_cfg: dict) -> None:
    """Bridge auxiliary model/endpoint overrides (vision, approval, plugins); compression reads yaml."""
    _aux_bridged_keys = {"vision", "approval"}
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            _aux_bridged_keys.add(_entry["key"])
    except Exception:
        pass  # plugin discovery failure must not break startup; built-in bridging stays intact
    for _task_key in _aux_bridged_keys:
        _task_cfg = _auxiliary_cfg.get(_task_key, {})
        if not isinstance(_task_cfg, dict):
            continue
        _upper = _task_key.upper()
        _prov = str(_task_cfg.get("provider", "")).strip()
        if _prov and _prov != "auto":
            os.environ[f"AUXILIARY_{_upper}_PROVIDER"] = _prov
        for _field, _suffix in (("model", "MODEL"), ("base_url", "BASE_URL"), ("api_key", "API_KEY")):
            _value = str(_task_cfg.get(_field, "")).strip()
            if _value:
                os.environ[f"AUXILIARY_{_upper}_{_suffix}"] = _value


def _bridge_config_to_env(_cfg: dict) -> None:
    """Export config.yaml settings to the env vars os.getenv() consumers read."""
    for _key, _val in _cfg.items():  # top-level scalars: fallback only, never override .env
        if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
            os.environ[_key] = str(_val)
    _terminal_cfg = _cfg.get("terminal", {})
    if _terminal_cfg and isinstance(_terminal_cfg, dict):
        _bridge_terminal_config_to_env(_terminal_cfg)
    _auxiliary_cfg = _cfg.get("auxiliary", {})
    if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
        _bridge_auxiliary_config_to_env(_auxiliary_cfg)
    # config.yaml is the documented, authoritative source for these settings — it unconditionally wins over
    # .env values. Previously the guards below read `if X not in os.environ` and let stale .env entries
    # (e.g. HERMES_MAX_ITERATIONS=60 written by an old `hermes setup` run) silently shadow the user's
    # current config. See PR #18413 / the 60-vs-500 max_turns incident.
    _agent_cfg = _cfg.get("agent", {})
    _bridge_max_turns_to_env(_agent_cfg)
    _bridge_section_to_env(_agent_cfg, _AGENT_ENV_BRIDGE)
    _bridge_section_to_env(_cfg.get("sessions", {}), _SESSIONS_ENV_BRIDGE)
    _display_cfg = _cfg.get("display", {})
    _bridge_section_to_env(_display_cfg, _DISPLAY_ENV_BRIDGE)
    # Documented service-manager override: env wins when set (other display bridges stay config-first).
    if (isinstance(_display_cfg, dict) and "busy_steer_ack_enabled" in _display_cfg
            and "HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED" not in os.environ):
        os.environ["HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED"] = str(_display_cfg["busy_steer_ack_enabled"])
    _tz_cfg = _cfg.get("timezone", "")
    if _tz_cfg and isinstance(_tz_cfg, str):
        os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
    _security_cfg = _cfg.get("security", {})
    if isinstance(_security_cfg, dict) and _security_cfg.get("redact_secrets") is not None:
        os.environ["HERMES_REDACT_SECRETS"] = str(_security_cfg["redact_secrets"]).lower()
    # Media policy uses the shared bridge so standalone entrypoints (`hermes cron run`) match.
    _gateway_cfg = _cfg.get("gateway", {})
    if isinstance(_gateway_cfg, dict):
        from gateway.media_policy import apply_media_policy_env
        apply_media_policy_env(_cfg)
        _trust_recent_seconds = _gateway_cfg.get("trust_recent_files_seconds")
        if _trust_recent_seconds is not None:
            os.environ["HERMES_MEDIA_TRUST_RECENT_SECONDS"] = str(_trust_recent_seconds)
        # platform_connect_timeout is an escape hatch, unlike the bridges above: env WINS if already set.
        if ("platform_connect_timeout" in _gateway_cfg
                and not os.environ.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()):
            os.environ["HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT"] = str(_gateway_cfg["platform_connect_timeout"])


def _load_bridge_config(config_path: Path) -> dict:
    """Raw config read for the presence-sensitive env bridge, with the managed overlay applied. Raw (not
    defaults-merged) so only keys the user wrote are bridged, else all of DEFAULT_CONFIG would be
    exported; the overlay applies BEFORE bridging so pinned values win in env too."""
    from hermes_cli.config import _expand_env_vars, read_user_config_raw
    cfg = _expand_env_vars(read_user_config_raw(config_path))
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        from hermes_cli import managed_scope
        cfg = managed_scope.apply_managed_overlay(cfg)
    except Exception:
        pass
    return cfg


_config_path = _hermes_home / 'config.yaml'
_cfg: dict = {}
if _config_path.exists():
    try:
        _cfg = _load_bridge_config(_config_path)
        _bridge_config_to_env(_cfg)
    except Exception as _bridge_err:
        # stderr, not logger: the module logger is not initialized yet at import time.
        print(
            f"  Warning: config.yaml → env bridge failed: {type(_bridge_err).__name__}: {_bridge_err}",
            file=sys.stderr)
        print(
            "  Gateway will fall back to .env values, which may not match "
            "your current config.yaml. Run `hermes doctor` to investigate.",
            file=sys.stderr)

# IPv4 preference must apply before any HTTP clients are created.
try:
    from hermes_constants import apply_ipv4_preference
    _network_cfg = _cfg.get("network", {})
    if isinstance(_network_cfg, dict) and _network_cfg.get("force_ipv4"):
        apply_ipv4_preference(force=True)
except Exception as _bootstrap_exc:
    print(f"  Warning: IPv4 preference application failed: {_bootstrap_exc}", file=sys.stderr)

try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception as _bootstrap_exc:
    print(f"  Warning: config validation failed: {_bootstrap_exc}", file=sys.stderr)

try:
    from hermes_cli.config import warn_deprecated_cwd_env_vars
    warn_deprecated_cwd_env_vars()
except Exception as _bootstrap_exc:
    print(f"  Warning: deprecation check failed: {_bootstrap_exc}", file=sys.stderr)

os.environ["HERMES_QUIET"] = "1"  # gateway runs quiet: no debug output, cwd used directly

# HERMES_EXEC_ASK is set in start_gateway(), NOT at import: CLI tools importing this module must not
# flip interactive sessions into ask-mode (approval prompts would become silent pending_approval).

# Terminal cwd: config.yaml terminal.cwd is canonical (bridged above); MESSAGING_CWD is legacy fallback.
from gateway.cwd_placeholder import CWD_PLACEHOLDERS, resolve_placeholder_terminal_cwd

_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in CWD_PLACEHOLDERS:
    _resolved_cwd = resolve_placeholder_terminal_cwd(
        configured_cwd=_configured_cwd,
        terminal_backend=os.environ.get("TERMINAL_ENV", ""),
        messaging_cwd=os.getenv("MESSAGING_CWD"),
        docker_mount_cwd_to_workspace=os.getenv(
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower()
        in {"true", "1", "yes"},
        home_fallback=str(Path.home()))
    if _resolved_cwd is None:
        os.environ.pop("TERMINAL_CWD", None)
    else:
        os.environ["TERMINAL_CWD"] = _resolved_cwd

from gateway.config import (
    ChannelOverride, Platform, GatewayConfig, PlatformConfig, _getenv, load_gateway_config)
from gateway.session import (
    AsyncSessionStore, SessionStore, SessionSource, SessionContext, build_session_key)
# Telegram topic routing (#22773, regression fixed #52060): a
# ``telegram:<positive_chat_id>:<numeric_thread_id>`` cron target is ambiguous — a forum-style topic in a
# private chat and a genuine Bot API channel Direct-Messages topic share the same shape and need OPPOSITE
# routing. Disambiguate at delivery time via ``_is_channel_dm_topic`` (see its docstring for the full
# rationale); ``thread_id`` goes in ``route_metadata`` so the anchorless cron send bypasses the
# DeliveryRouter's private-chat reply-anchor requirement. Compute the routed metadata ONCE so both the text
# send (via DeliveryRouter) and the media send agree.
from gateway.delivery import DeliveryRouter
from gateway.turn_lease import SessionTurnLeaseRegistry
from gateway.session_state import SessionState, legacy_dict_property, legacy_lease_token_property
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.turn_context import TurnContext
from gateway.run_voice import GatewayVoiceMixin
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_topics import GatewayTopicThreadsMixin
from gateway.run_turn import GatewayTurnMixin, is_context_overflow_failure_result
from gateway.run_shutdown import GatewayShutdownMixin, _exit_with_failure_verdict, _resolve_gateway_exit_verdict
from gateway.run_busy import GatewayBusySessionMixin
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from gateway.run_startup import GatewayStartupMixin
from gateway.run_watchers import GatewaySessionWatchersMixin
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_goals import GatewayGoalsMixin
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.platforms.base import (
    BasePlatformAdapter,
    _reply_anchor_for_event,
    build_auto_tts_output_path,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT)


logger = logging.getLogger(__name__)


def _best_effort(fn: Callable[[], Any], debug_msg: Optional[str] = None) -> Any:
    """Call ``fn``; return None on any Exception (debug-logged via ``debug_msg`` ``%s`` if given)."""
    try:
        return fn()
    except Exception as exc:
        if debug_msg:
            logger.debug(debug_msg, exc)
        return None


# Shutdown quiesce ceiling for the gateway-owned thread pool. Drain already waited for the agents; what
# remains is short blocking work; anything slower is a stuck worker not worth waiting on (leash-clamped).
_EXECUTOR_QUIESCE_TIMEOUT = 2.0


_OWN_POLICY_OPEN_ENV = {
    Platform.WECOM: ("WECOM_DM_POLICY", "WECOM_GROUP_POLICY", "WECOM_ALLOW_ALL_USERS"),
    Platform.WEIXIN: ("WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY", "WEIXIN_ALLOW_ALL_USERS"),
    Platform.YUANBAO: ("YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY", "YUANBAO_ALLOW_ALL_USERS"),
    Platform.QQBOT: (None, None, "QQ_ALLOW_ALL_USERS"),
    Platform.WHATSAPP: ("WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "WHATSAPP_ALLOW_ALL_USERS")}


def _own_policy_open_startup_violation(config) -> Optional[str]:
    """Return a startup-abort reason when open policy lacks allow-all opt-in."""
    for platform, platform_config in getattr(config, "platforms", {}).items():
        if not getattr(platform_config, "enabled", False):
            continue
        open_env = _OWN_POLICY_OPEN_ENV.get(platform)
        if not open_env:
            continue
        dm_env, group_env, allow_all_env = open_env
        extra = getattr(platform_config, "extra", None) or {}
        dm_policy = str(extra.get("dm_policy")
                        or (_getenv(dm_env, "pairing") if dm_env else "pairing")).strip().lower()
        group_policy = str(
            extra.get("group_policy") or (_getenv(group_env, "pairing") if group_env else "pairing")
        ).strip().lower()
        if dm_policy != "open" and group_policy != "open":
            continue
        gateway_allow_all = _getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
        if gateway_allow_all or (
                allow_all_env and _getenv(allow_all_env, "").lower() in {"true", "1", "yes"}):
            continue
        return f"{platform.value}: open policy without allow-all opt-in"
    return None


# Placed into _running_agents *before* any await so a second message can't slip past the "already
# running" guard before the agent exists.
_AGENT_PENDING_SENTINEL = object()

# Conversation-scoped per-session state registry (legacy contract). State lives in
# ``SessionState.conversation`` (cleared via ``ConversationState.clear()``); this list remains for
# plain-dict stores not yet folded in (``_pending_model_notes``, popped per-key by
# _clear_conversation_scope) and the public test contract. NOT listed (different lifecycles): turn-scoped
# _running_agents*/_active_session_leases/_busy_ack_ts/_turn_lease_tokens (_release_running_agent_state +
# dispatch finally); _session_run_generation (monotonic; clearing breaks stale-run detection);
# _agent_cache (_evict_cached_agent); approval/slash-confirm (_clear_session_boundary_security_state).
# The state itself now lives in ``SessionState.conversation`` (see gateway/session_state.py) and boundaries
# clear it structurally via ``ConversationState.clear()`` — adding a field to ConversationState means every
# boundary picks it up automatically. History: boundaries used to each carry a hand-copied pop-list that
# drifted whenever a new dict was added (#48031, #58403, #10702, #35809). - _agent_cache: has its own
# eviction path (_evict_cached_agent) with resource cleanup; boundaries call it explicitly.
_CONVERSATION_SCOPED_STATE: tuple = (
    "_session_model_overrides",
    "_pending_one_turn_model_restores",
    "_session_reasoning_overrides",
    "_session_service_tier_overrides",
    "_pending_model_notes",
    "_last_resolved_model",
    "_queued_events",
    # Stall-watchdog "already notified" latch; cleared on /new so a fresh conversation can warn again.
    # See #72016.
    "_session_stall_notified",
    # Sidecar notes staged but never consumed (turn aborted before run_sync) must not leak into a
    # future conversation's first user message — session keys are source-derived and REUSED.
    "_pending_turn_sidecar_notes")


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances.
    ``resolve_runtime_provider()`` may fall back to env vars; behavioral config is config.yaml only."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider, format_runtime_provider_error, _get_model_config)
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    try:
        runtime = resolve_runtime_provider()
    except AuthError as auth_exc:
        # Rate-limit cap vs real auth failure: both use the fallback chain; the log must not mislabel.
        # Distinguish a transient rate-limit/quota cap (credentials are fine, re-auth cannot help) from a
        # genuine auth failure (expired/revoked token). See #32790.
        if is_rate_limited_auth_error(auth_exc):
            logger.warning("Primary provider rate-limited (429): %s — trying fallback", auth_exc)
        else:
            logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc


    capabilities = runtime.get("capabilities")
    capabilities = (
        {k: v for k, v in capabilities.items() if isinstance(k, str) and isinstance(v, bool)}
        if isinstance(capabilities, dict) else {})

    return {**_runtime_agent_kwargs(runtime), "capabilities": capabilities}


def _runtime_agent_kwargs(runtime: dict) -> dict:
    """AIAgent constructor kwargs shared by every runtime-provider resolution.
    ``request_overrides`` passes through as resolved so the provider's request body reaches each turn."""
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "requested_provider": runtime.get("requested_provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "request_overrides": runtime.get("request_overrides")}


@dataclasses.dataclass(frozen=True)
class _GatewayModelContext:
    """Effective gateway model route and context-window resolution."""

    model: str
    provider: str
    base_url: str
    context_length: int
    context_source: str


def _resolve_gateway_model_context(model: Optional[str] = None) -> _GatewayModelContext:
    """Resolve the configured gateway route and effective context window. Call off-loop (may block)."""
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT, get_model_context_length
    resolved_model = model or _resolve_gateway_model()
    config_context_length = provider = base_url = api_key = custom_providers = None
    configured_model = configured_provider = configured_base_url = None

    def _read_config() -> None:
        nonlocal config_context_length, provider, base_url, custom_providers
        nonlocal configured_model, configured_provider, configured_base_url
        data = _load_gateway_config()
        if not data:
            return
        model_cfg = data.get("model", {})
        if isinstance(model_cfg, dict):
            configured_model = model_cfg.get("default") or model_cfg.get("model")
            raw_ctx = model_cfg.get("context_length")
            if raw_ctx is not None:
                with suppress(TypeError, ValueError):
                    config_context_length = int(raw_ctx)
            configured_provider = provider = model_cfg.get("provider") or None
            configured_base_url = base_url = model_cfg.get("base_url") or None
        try:
            from hermes_cli.config import get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(data)
        except Exception:
            custom_providers = data.get("custom_providers")

    def _read_runtime() -> None:
        nonlocal provider, base_url, api_key
        runtime = _resolve_runtime_agent_kwargs()
        provider = runtime.get("provider") or provider
        base_url = runtime.get("base_url") or base_url
        api_key = runtime.get("api_key")

    def _pin_still_applies() -> bool:
        # Drop a configured context_length pin when the effective route no longer matches (or on error).
        from hermes_cli.route_identity import should_clear_context_pin
        return not should_clear_context_pin(
            configured_model, resolved_model, configured_base_url, base_url, configured_provider, provider)

    def _custom_ctx() -> Optional[int]:
        from hermes_cli.config import get_custom_provider_context_length
        return get_custom_provider_context_length(
            model=resolved_model, base_url=base_url, custom_providers=custom_providers)

    _best_effort(_read_config)
    _best_effort(_read_runtime)
    if config_context_length is not None and not _best_effort(_pin_still_applies):
        config_context_length = None
    if config_context_length is None and custom_providers and base_url:
        config_context_length = _best_effort(_custom_ctx) or None

    context_length = get_model_context_length(
        resolved_model, base_url=base_url or "", api_key=api_key or "",
        config_context_length=config_context_length, provider=provider or "",
        custom_providers=custom_providers)
    context_source = ("config" if config_context_length is not None
                      else "default" if context_length == DEFAULT_FALLBACK_CONTEXT else "detected")
    return _GatewayModelContext(
        model=resolved_model, provider=provider or "", base_url=base_url or "",
        context_length=context_length, context_source=context_source)


def _resolve_runtime_agent_kwargs_for_provider(provider: str) -> dict:
    """Resolve runtime credentials for a specific provider (e.g. from channel override)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
    try:
        runtime = resolve_runtime_provider(requested=provider)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc
    return {
        **_runtime_agent_kwargs(runtime),
        "request_overrides": dict(runtime.get("request_overrides") or {}),
        "capabilities": dict(runtime.get("capabilities") or {})}


def _deep_merge_request_overrides(base: Optional[dict], override: Optional[dict]) -> dict:
    """Merge request_overrides dicts, deep-merging nested dictionaries."""
    from hermes_cli.config import _deep_merge
    base_dict = dict(base or {})
    override_dict = dict(override or {})
    if not base_dict:
        return override_dict
    if not override_dict:
        return base_dict
    return _deep_merge(base_dict, override_dict)


def _credential_pool_for_provider(provider: Optional[str]):
    """Return the live credential pool for a provider id (e.g. ``custom:hyper``)."""
    if not provider or not str(provider).strip():
        return None
    try:
        return _resolve_runtime_agent_kwargs_for_provider(str(provider).strip()).get("credential_pool")
    except Exception:
        logger.debug("Failed to resolve credential pool for provider=%s", provider, exc_info=True)
        return None


def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_model/fallback_providers config."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        # Canonical loader so managed overlay / ${VAR} expansion reach the fallback chain.
        cfg = _load_gateway_runtime_config()
        fb_list = get_fallback_chain(cfg)
        if not fb_list:
            return None
        for entry in fb_list:
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key
                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"), explicit_base_url=entry.get("base_url"),
                    explicit_api_key=resolve_entry_api_key(entry))
                # Log the config `provider`, not the runtime category (Ollama would log "openrouter").
                logger.info(
                    # Log the literal `provider` key from config, not the resolved runtime category — an
                    # Ollama fallback resolves through the OpenAI-compatible path and would otherwise be
                    # logged as "openrouter", contradicting the operator's config (#32790).
                    "Fallback provider resolved: %s model=%s",
                    entry.get("provider") or runtime.get("provider"), entry.get("model"))
                return {**_runtime_agent_kwargs(runtime), "model": entry.get("model")}
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
    except Exception:
        pass
    return None


def _event_media_type_at(event, index: int) -> str:
    """Per-attachment MIME at *index*; "" when the adapter set only a message-level type."""
    media_types = getattr(event, "media_types", None) or []
    return media_types[index] if index < len(media_types) else ""


def _event_media_kind_is(event, index: int, mime_prefix: str, fallback_types: frozenset) -> bool:
    """Per-attachment MIME first, message-level type only when unknown (else a document uploaded
    alongside an image is base64'd as vision and the provider 400s)."""
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith(mime_prefix)
    return getattr(event, "message_type", None) in fallback_types


def _event_media_is_image(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "image/", frozenset({MessageType.PHOTO}))


def _event_media_is_audio(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "audio/", frozenset({MessageType.VOICE, MessageType.AUDIO}))


def _event_media_is_stt_input(event, index: int) -> bool:
    """True when an audio attachment should enter the automatic STT pipeline."""
    message_type = getattr(event, "message_type", None)
    if message_type in {MessageType.AUDIO, MessageType.DOCUMENT}:
        return False
    return message_type == MessageType.VOICE or _event_media_type_at(event, index).startswith("audio/")


def _event_media_is_video(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "video/", frozenset({MessageType.VIDEO}))


def _build_media_placeholder(event) -> str:
    """Text placeholder for media-only events (later replaced by vision enrichment).
    Queued media is dequeued via .text only, so a caption-less event would otherwise be lost."""
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    for i, url in enumerate(media_urls):
        if _event_media_is_image(event, i):
            parts.append(f"[User sent an image: {url}]")
        elif _event_media_is_audio(event, i):
            parts.append(f"[User sent audio: {url}]")
        elif _event_media_is_video(event, i):
            parts.append(f"[User sent a video: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)


def _build_document_context_note(
    display_name: str, agent_path: str, mtype: str, *, content_inlined: bool = True) -> str:
    """Context note prepended to a user turn when they attach a document.
    ``content_inlined=False`` = cached without content, so tell the agent to read it. Binary docs must
    say *extract* the text; "ask the user" made it punt."""
    if mtype.startswith("text/") and content_inlined:
        return (
            f"[The user sent a text document: '{display_name}'. Its content has been included below. "
            f"The file is also saved at: {agent_path}]")
    if mtype.startswith("text/"):
        return (
            f"[The user sent a text document: '{display_name}'. It is saved at: {agent_path}. "
            f"Its content is not inlined here. Read the cached file yourself before answering "
            f"when the user's request involves its contents.]")
    return (
        f"[The user sent a document: '{display_name}'. It is saved at: {agent_path}. "
        f"Its text is not inlined here (it's a binary format such as PDF or DOCX). "
        f"To read it, extract the document's text yourself — for example with the "
        f"terminal tool or the ocr-and-documents skill — before answering, instead "
        f"of asking the user to paste the contents.]")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


async def _probe_audio_duration(path: str) -> Optional[str]:
    """Best-effort duration probe. Returns formatted MM:SS / HH:MM:SS, or None on failure."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        try:
            def _wav_duration() -> float:
                import wave
                with wave.open(path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 1
                    return frames / float(rate)
            return _format_duration(await asyncio.to_thread(_wav_duration))
        except Exception:
            pass
    if ext in (".ogg", ".opus", ".oga"):
        try:
            def _ogg_duration() -> float:
                from mutagen.oggopus import OggOpus
                return float(OggOpus(path).info.length)
            return _format_duration(await asyncio.to_thread(_ogg_duration))
        except Exception:
            pass
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            return _format_duration(float(stdout.decode().strip()))
    except Exception:
        pass

    return None


def _dequeue_pending_event(adapter, session_key: str) -> MessageEvent | None:
    """Consume and return the pending event; media metadata is kept so follow-ups re-enter preprocessing.
    """
    return adapter.get_pending_message(session_key)


_INTERRUPT_REASON_STOP = "Stop requested"
_INTERRUPT_REASON_RESET = "Session reset requested"
_INTERRUPT_REASON_TIMEOUT = "Execution timed out (inactivity)"
_INTERRUPT_REASON_SSE_DISCONNECT = "SSE client disconnected"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"
_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"


def _reap_gateway_turn_processes(
    task_id: str, process_baseline, *, source: str,
    is_still_current: Optional[Callable[[], bool]] = None) -> int:
    """Reap only background processes created by one abandoned turn.
    ``task_id`` is session-scoped, so a *replacement* turn can spawn its own process mid-reap;
    ``is_still_current`` lets the caller bail instead of killing it (that turn owns its own baseline)."""
    if not task_id:
        # Blank task_id (sessionless callers) would match and kill every unrelated empty-task process.
        return 0
    if is_still_current is not None:
        try:
            if not is_still_current():
                logger.debug(
                    "Skipping reap for turn %s (%s): a newer turn already "
                    "claimed this session; it owns its own baseline.", task_id, source)
                return 0
        except Exception:
            logger.debug(
                "is_still_current check failed for turn %s (%s); reaping anyway",
                task_id, source, exc_info=True)

    from tools.process_registry import process_registry
    try:
        killed = process_registry.kill_started_since(task_id, process_baseline, source=source)
    except Exception:
        # Detached daemon thread: an uncaught exception would only reach threading.excepthook.
        logger.warning(
            "Failed to reap background processes for turn %s (%s)", task_id, source, exc_info=True)
        return 0
    if killed:
        logger.warning(
            "Reaped %d background process(es) created by abandoned turn %s (%s)",
            killed, task_id, source)
    return killed


_TURN_STACK_DUMP_FRAME_MARKERS = (
    "run_conversation", "run_sync", "_run_sync_with_timeout_lifecycle",
    "finalize_turn", "end_turn", "run_in_session")


def _dump_wedged_turn_stacks(task_id: str) -> None:
    """Log the stack of every thread that looks like turn work, at reap time.
    The hard interrupt frees the wedged worker before a profiler can attach, so dump BEFORE it.
    Best-effort, bounded (turn-machinery threads only, capped output), never raises."""
    try:
        frames = sys._current_frames()
        names = {t.ident: t.name for t in threading.enumerate()}
        dumped = 0
        for ident, frame in frames.items():
            if ident == threading.get_ident():
                continue  # the reaper itself
            stack = traceback.format_stack(frame)
            joined = "".join(stack)
            if not any(marker in joined for marker in _TURN_STACK_DUMP_FRAME_MARKERS):
                continue
            dumped += 1
            if dumped > 8:
                logger.error(
                    "Wedged-turn stack dump for task %s truncated: more than 8 candidate threads",
                    task_id)
                break
            logger.error(
                "Wedged-turn stack dump (task=%s thread=%s ident=%s):\n%s",
                task_id, names.get(ident, "?"), ident, "".join(stack[-25:]))
        if dumped == 0:
            logger.error(
                "Wedged-turn stack dump for task %s: no thread with "
                "turn-machinery frames found (worker may have already exited)", task_id)
    except Exception:
        logger.debug("Wedged-turn stack dump failed", exc_info=True)


def _abandon_timed_out_gateway_turn(
    *, agent_holder, task_id: str, process_baseline, worker_done: threading.Event,
    timeout_fired: threading.Event, cleanup_lock: threading.Lock,
    is_still_current: Optional[Callable[[], bool]] = None) -> bool:
    """Interrupt one timed-out turn and reap only processes it created."""
    with cleanup_lock:
        if worker_done.is_set() or timeout_fired.is_set():
            return False
        timeout_fired.set()

    # BEFORE interrupting: the interrupt frees the blocked frame, destroying the only evidence.
    _dump_wedged_turn_stacks(task_id)

    agent = agent_holder[0] if agent_holder else None
    if agent is not None:
        try:
            request_hard_interrupt(agent, _INTERRUPT_REASON_TIMEOUT)
        except Exception:
            logger.debug("Timed-out agent interrupt failed", exc_info=True)

    try:
        _reap_gateway_turn_processes(
            task_id, process_baseline, source="gateway_turn_timeout",
            is_still_current=is_still_current)
    except Exception:
        logger.warning(
            "Failed to reap background processes for timed-out turn %s", task_id, exc_info=True)
    return True


def _watch_gateway_turn_inactivity(
    *, agent_holder, task_id: str, process_baseline, timeout: float, worker_done: threading.Event,
    timeout_fired: threading.Event, cleanup_lock: threading.Lock, poll_interval: float = 5.0,
    is_still_current: Optional[Callable[[], bool]] = None) -> None:
    """Thread watchdog that remains runnable when gateway asyncio is starved."""
    while not worker_done.wait(max(0.01, poll_interval)):
        agent = agent_holder[0] if agent_holder else None
        if agent is None or not hasattr(agent, "get_activity_summary"):
            continue
        try:
            idle_seconds = float(agent.get_activity_summary().get("seconds_since_activity", 0.0))
        except Exception:
            continue
        if idle_seconds < timeout:
            continue
        _abandon_timed_out_gateway_turn(
            agent_holder=agent_holder, task_id=task_id, process_baseline=process_baseline,
            worker_done=worker_done, timeout_fired=timeout_fired, cleanup_lock=cleanup_lock,
            is_still_current=is_still_current)
        return


_CONTROL_INTERRUPT_MESSAGES = frozenset({
    _INTERRUPT_REASON_STOP.lower(), _INTERRUPT_REASON_RESET.lower(),
    _INTERRUPT_REASON_TIMEOUT.lower(), _INTERRUPT_REASON_SSE_DISCONNECT.lower(),
    _INTERRUPT_REASON_GATEWAY_SHUTDOWN.lower(), _INTERRUPT_REASON_GATEWAY_RESTART.lower()})


def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    return " ".join(str(message).strip().split()).lower() in _CONTROL_INTERRUPT_MESSAGES


def _strip_response_attachments_for_direct_send(response: str, adapter) -> str:
    """Return the visible text portion of a response before direct send().
    Only explicit ``MEDIA:`` attachments are stripped; bare paths/URLs stay visible. No broad regex after
    ``extract_media()``: it deliberately preserves protected code spans and unvalidated tags.

    Queued follow-up resends only replay explicit ``MEDIA:`` attachments in this path. Keep bare local paths
    and ordinary image URLs visible because the post-stream uploader intentionally ignores them (#20834).
    """
    _, cleaned = adapter.extract_media(response)
    return cleaned.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "").strip()


def _skill_slug_from_frontmatter(skill_md: Path) -> tuple[str | None, str | None]:
    """Derive ``(slug, declared_name)`` from a SKILL.md; ``(None, None)`` if unreadable or no ``name:``.
    Matches ``scan_skill_commands``: the slug comes from frontmatter ``name:``, NOT the directory."""
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    content = content.lstrip("\ufeff")  # tolerate UTF-8 BOM (Windows editors)
    if not content.startswith("---"):
        return None, None
    end = content.find("\n---", 3)
    if end < 0:
        return None, None
    declared_name: str | None = None
    for line in content[3:end].splitlines():
        line = line.strip()
        if line.startswith("name:"):
            raw = line.split(":", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            declared_name = raw.strip()
            break
    if not declared_name:
        return None, None
    slug = declared_name.lower().replace(" ", "-").replace("_", "-")
    # Mirrors _SKILL_INVALID_CHARS / _SKILL_MULTI_HYPHEN from skill_commands
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return (slug or None), declared_name


def _check_unavailable_skill(command_name: str) -> str | None:
    """Hint when a command matches a skill that is disabled or optional-install only; else None."""
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
        disabled = _get_disabled_skill_names()

        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, declared_name = _skill_slug_from_frontmatter(skill_md)
                if not slug or not declared_name:
                    continue
                # disabled is keyed by the declared frontmatter name (what skills.disabled stores).
                if slug == normalized and declared_name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`")

        # Check optional skills (shipped with repo but not installed)
        from hermes_constants import get_optional_skills_dir
        repo_root = Path(__file__).resolve().parent.parent
        optional_dir = get_optional_skills_dir(repo_root / "optional-skills")
        if optional_dir.exists():
            for skill_md in optional_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, _declared = _skill_slug_from_frontmatter(skill_md)
                if not slug or slug != normalized:
                    continue
                # Install path: official/<category>/<name>
                rel = skill_md.parent.relative_to(optional_dir)
                install_path = f"official/{'/'.join(rel.parts)}"
                return (
                    f"The **{command_name}** skill is available but not installed.\n"
                    f"Install it with: `hermes skills install {install_path}`")
    except Exception:
        pass
    return None


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value


def _teams_pipeline_plugin_enabled() -> bool:
    """Return True when the standalone Teams pipeline plugin is enabled."""
    enabled = cfg_get(_load_gateway_config(), "plugins", "enabled", default=[])
    return isinstance(enabled, list) and ("teams_pipeline" in enabled or "teams-pipeline" in enabled)


def _gateway_config_home() -> Path:
    """Return the Hermes home that gateway config reads should use."""
    override = get_hermes_home_override()
    return Path(override) if override else _hermes_home


def _load_gateway_config(config_path: "Path | None" = None) -> dict:
    """Load and parse a gateway config.yaml, returning {} on any error (fail-open).
    Defaults to the active gateway home (``_hermes_home`` monkeypatches apply); multiplexers pass a path.
    """
    if config_path is None:
        config_path = _gateway_config_home() / 'config.yaml'
    raw: dict = {}
    used_canonical = False
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        # Fast path via shared cache when the path is canonical; else direct read (test monkeypatches).
        if config_path == get_config_path():
            raw = read_raw_config()
            used_canonical = True
    except Exception:
        pass

    if not used_canonical:
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load gateway config from %s", config_path)
            raw = {}

    # Neither read_raw_config() nor yaml.safe_load carries the managed merge; overlay on both paths.
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    if not isinstance(raw, dict):
        return {}
    # Canonicalize model-id aliases (model.name/model.model → model.default) and migrate stale root
    # provider/base_url: the gateway bypasses load_config(), else ``model: {name: <id>}`` is empty.
    try:
        # The gateway bypasses load_config() (it reads raw YAML for speed), so the normalization that
        # load_config() applies must be replayed here or the gateway would resolve an empty model for
        # ``model: {name: <id>}`` configs while the CLI resolves it correctly. See issue #34500. Fail-open.
        from hermes_cli.config import _normalize_root_model_keys
        raw = _normalize_root_model_keys(raw)
    except Exception:
        pass
    return raw


def _checkpoint_agent_kwargs(config: dict | None) -> dict:
    """Translate gateway checkpoint config into ``AIAgent`` constructor args.
    Gateway bypasses ``load_config()``, so defaults are here; legacy ``checkpoints: true`` works."""
    cp_cfg = config.get("checkpoints", {}) if isinstance(config, dict) else {}
    if isinstance(cp_cfg, bool):
        cp_cfg = {"enabled": cp_cfg}
    elif not isinstance(cp_cfg, dict):
        cp_cfg = {}
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["checkpoints"]
    return {
        "checkpoints_enabled": cp_cfg.get("enabled", defaults["enabled"]),
        "checkpoint_max_snapshots": cp_cfg.get("max_snapshots", defaults["max_snapshots"]),
        "checkpoint_max_total_size_mb": cp_cfg.get("max_total_size_mb", defaults["max_total_size_mb"]),
        "checkpoint_max_file_size_mb": cp_cfg.get("max_file_size_mb", defaults["max_file_size_mb"])}


def _load_gateway_runtime_config() -> dict:
    """Load gateway config for runtime reads, expanding supported ``${VAR}`` refs.
    Expansion failures are deliberately NOT swallowed: an unexpanded dict would mask the bug fixed here.
    """
    cfg = _load_gateway_config()
    if not isinstance(cfg, dict) or not cfg:
        return {}
    from hermes_cli.config import _expand_env_vars
    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else {}


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml (single source of truth), else temporary AIAgents (e.g. /compress)
    use the hardcoded default, which fails under openai-codex."""
    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _channel_override_lookup_keys(
    chat_id: str, *, thread_id: Optional[str] = None, parent_id: Optional[str] = None) -> list[str]:
    """Ordered, de-duplicated ``channel_overrides`` lookup keys (matches ``resolve_channel_prompt``:
    exact id first, then parent — Discord threads inherit parent overrides)."""
    return list(dict.fromkeys(str(key) for key in (chat_id, thread_id, parent_id) if key))


def _get_channel_override(
    config: GatewayConfig, platform: Platform, chat_id: str, *, thread_id: Optional[str] = None,
    parent_id: Optional[str] = None) -> Optional[ChannelOverride]:
    """Per-channel override via chat_id, then thread_id, then parent_id; None if absent."""
    platforms = getattr(config, "platforms", None)
    if not platforms:
        return None
    platform_config = platforms.get(platform)
    if not platform_config or not platform_config.channel_overrides:
        return None
    overrides = platform_config.channel_overrides
    for key in _channel_override_lookup_keys(chat_id, thread_id=thread_id, parent_id=parent_id):
        ov = overrides.get(key)
        if ov is not None:
            return ov
    return None


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Hermes update command argv: ``hermes`` on PATH, else ``python -m hermes_cli.main``, else None."""
    import shutil
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]
    try:
        import importlib.util
        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass
    return None


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key (``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``).
    For group/channel sessions the suffix may be a user_id, not a thread_id, so ``thread_id`` is omitted.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {"platform": parts[2], "chat_type": parts[3], "chat_id": parts[4]}
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None


def _shorten_command_for_display(command: str, limit: int = 80) -> str:
    """Collapse a shell command onto one line and cap its length for display."""
    one_line = " ".join((command or "").split())
    if len(one_line) > limit:
        one_line = one_line[: limit - 1] + "…"
    return one_line


def _format_concise_process_notification(
    session_id: str, command: str, exit_code, output: str, duration_seconds=None) -> str:
    """One-line completion message for ``concise`` display mode; failure appends a short output tail."""
    ok = exit_code in {0, None}
    icon = "✅" if ok else "❌"
    verb = "finished" if ok else f"failed (exit {exit_code})"
    parts = [f"{icon} Background task {verb}"]
    short_cmd = _shorten_command_for_display(command)
    if short_cmd:
        parts.append(f"— `{short_cmd}`")
    if isinstance(duration_seconds, (int, float)) and duration_seconds >= 0:
        secs = int(duration_seconds)
        if secs >= 3600:
            dur = f"{secs // 3600}h {(secs % 3600) // 60}m"
        elif secs >= 60:
            dur = f"{secs // 60}m {secs % 60}s"
        else:
            dur = f"{secs}s"
        parts.append(f"({dur})")
    text = " ".join(parts)
    if not ok and output:
        tail_lines = [ln for ln in output.strip().splitlines() if ln.strip()][-5:]
        tail = "\n".join(tail_lines)
        if len(tail) > 500:
            tail = tail[-500:]
        if tail:
            text += f"\n```\n{tail}\n```"
    return text


def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    # watch_disabled / overflow events carry their summary in `message` (process_registry formatter).
    if evt_type in ("watch_disabled", "watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\nMatched output:\n{_out}")
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        from tools.process_registry_notifications import format_process_notification
        return format_process_notification(evt)

    return None


def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.
    Foreign events requeued inside ``while not queue.empty()`` never terminate: detach, then requeue."""
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {
            "watch_match", "watch_disabled", "watch_overflow_tripped", "watch_overflow_released"}:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events


# Weak ref to the active GatewayRunner; tools like send_message route through its live adapters.
import weakref as _weakref
_gateway_runner_ref: _weakref.ref = lambda: None


def _normalize_empty_agent_response(
    agent_result: dict, response: str, *, history_len: int = 0) -> str:
    """Normalize empty/None agent responses into user-facing messages.
    Covers ``failed``, work done (api_calls > 0) with no text, and never-ran (api_calls == 0, the
    post-/stop silent-drop from a stale generation token) with a retry hint.

    Consolidates the existing ``failed`` handler and adds a catch-all for the case where the agent did work
    (api_calls > 0) but returned no text. Fix for #18765.
    Also surfaces a retry hint when the agent never ran at all (api_calls == 0) for a non-interrupted,
    non-failed turn -- this is the silent-drop pattern observed after ``/stop`` where the next user message
    hits a stale generation token and returns an empty result, leaving the platform with nothing to send.
    (#31884)

    A failed context-overflow turn whose ``final_response`` is only the raw provider envelope
    (``HTTP 400: {...}``) is rewritten too: returned unchanged, chat sanitizers turn it into a
    generic provider-failed reply and the user never sees /compact. Curated agent text survives.
    """
    is_overflow = is_context_overflow_failure_result(agent_result, history_len)
    if response and not (is_overflow and _looks_like_gateway_provider_error(response)):
        return response
    if agent_result.get("failed"):
        # ``error`` can be an EXPLICIT None (bypasses dict.get default) -> would render "failed: None".
        error_detail = agent_result.get("error") or "unknown error"
        error_str = str(error_detail).lower()
        # Persistence failures: suggesting /reset would destroy context without fixing storage.
        failure_reason = str(agent_result.get("failure_reason") or "")
        if failure_reason.startswith("session_persistence_failed") or "session storage" in error_str:
            if failure_reason.endswith(":disk") or "disk" in error_str:
                return (
                    "⚠️ Session storage was temporarily unavailable, so this "
                    "turn was stopped to protect your conversation history. "
                    "Please check available disk space, then send your message again.")
            return (
                "⚠️ Session storage was temporarily unavailable, so this "
                "turn was stopped to protect your conversation history. "
                "Your message should already be saved — please send it again in a moment.")
        if is_overflow:
            return (
                "⚠️ Session too large for the model's context window.\n"
                "Use /compact to compress the conversation, or /reset to start fresh.")
        return (
            f"The request failed: {str(error_detail)[:300]}\n"
            "Try again or use /reset to start a fresh session.")

    api_calls = int(agent_result.get("api_calls", 0) or 0)
    if agent_result.get("interrupted"):
        # Interrupted with api_calls > 0 = deliberately stopped/steered; silence is intentional (queued
        # messages arrive via the recursive drain). ZERO api_calls = never processed (stale /stop flag).
        # An interrupted run that did work (api_calls > 0) is the drain of a run the user deliberately
        # stopped or steered — its silence is intentional, and any queued/interrupting message is delivered
        # by the recursive drain inside _run_agent before this result is seen. An interrupted run with ZERO
        # api_calls never processed the user's message at all: it was killed at the top of the tool loop by
        # an interrupt flag left over from a recent /stop (#44212). Pure silence there swallows a real user
        # message, so surface it.
        # api_calls == 0, not failed, not interrupted: the agent never ran for this turn. This is the
        # post-/stop generation-race pattern where the gateway would otherwise silently drop the turn
        # (response=0 chars) and the user sees no reply at all. Surface a short retry hint so the message
        # isn't lost in silence. (#31884)
        if api_calls == 0:
            return (
                "⚠️ Your message was interrupted before processing started "
                "(likely by a recent /stop). Please send it again.")
        return response
    if api_calls > 0:
        # Hidden-reasoning-only retry exhaustion: the loop's sentinel text ("Codex response remained
        # incomplete after 3 continuation attempts") doubles as final_response, so it would be delivered
        # verbatim into the channel — where peer agents can ingest it as a completed assistant turn
        # (#51628). Blank it here so the normal empty-response handling (and the suppression below) applies.
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            return ""
        if agent_result.get("partial"):
            err = agent_result.get("error", "processing incomplete")
            return f"⚠️ Processing stopped: {str(err)[:200]}. Try again."
        return (
            "⚠️ Processing completed but no response was generated. "
            "This may be a transient error — try sending your message again.")

    # api_calls == 0, not failed/interrupted: agent never ran (post-/stop race); don't drop silently.
    if api_calls == 0 and not agent_result.get("partial"):
        return (
            "⚠️ Your message wasn't processed (the previous turn was still "
            "being cleaned up). Please send it again.")

    return response


def _is_gateway_hidden_reasoning_incomplete_turn(agent_result: dict) -> bool:
    """Detect retry-exhausted turns with hidden reasoning but no visible answer.
    The loop returns the retry-exhaustion sentinel as BOTH ``final_response`` and ``error``, so a
    non-empty ``final_response`` proves nothing; any text other than the sentinel is a real answer."""
    if (not isinstance(agent_result, dict) or agent_result.get("failed")
            or agent_result.get("interrupted") or not agent_result.get("partial")):
        return False
    error_text = str(agent_result.get("error", "") or "").strip()
    if "remained incomplete after" not in error_text.lower():
        return False
    final_response = str(agent_result.get("final_response") or "").strip()
    return not final_response or final_response == error_text


def _should_clear_resume_pending_after_turn(agent_result: dict) -> bool:
    """True only when a gateway turn really completed successfully.
    ``resume_pending`` is a durable restart-recovery marker; a soft interrupt can look like a normal
    empty result, and clearing then loses the signal."""
    if not isinstance(agent_result, dict) or agent_result.get("interrupted"):
        return False
    if agent_result.get("failed") or agent_result.get("partial") or agent_result.get("error"):
        return False
    return agent_result.get("completed") is not False


def _preserve_queued_followup_history_offset(
    current_result: dict, followup_result: dict) -> dict:
    """Carry the outer history offset through queued follow-up drains.
    Each recursive ``_run_agent()`` advances ``history_offset``; uncorrected, the outer persistence
    step sees only the *last* queued turn as "new" and drops earlier ones."""
    if not isinstance(followup_result, dict) or not isinstance(current_result, dict):
        return followup_result
    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if not isinstance(current_offset, int):
        return followup_result
    if isinstance(followup_offset, int) and followup_offset <= current_offset:
        return followup_result
    return {**followup_result, "history_offset": current_offset}


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters`` (may be ``None``).
    Nothing else calls ``disconnect()`` on it, so ``__init__`` resources (e.g. SQLite fds) would leak
    until GC (not prompt for asyncio-bound objects) and exhaust the fd ulimit over a long retry loop.

    The reconnect watcher in ``GatewayRunner._platform_reconnect_watcher`` constructs a fresh adapter on
    every retry attempt. When the connect call fails — for any of the three reasons (non-retryable error,
    retryable error, exception during connect) — the adapter is dropped without ever being installed, so
    nothing else will call its ``disconnect()``. ``APIServerAdapter`` opens a SQLite ``ResponseStore`` that
    holds 2 fds — the db file and its WAL sidecar) stay open until garbage collection sweeps the unreachable
    object, which Python's cyclic GC does not do promptly for asyncio-bound objects with native handles. The
    cumulative leak is 2 fds × every retry at the 300s backoff cap ≈ 12 fds/hour, and the default 2560-fd
    ulimit is exhausted in ~12h of continuous failure, after which every open() call on the gateway raises
    ``OSError: [Errno 24] Too many open files`` and the gateway becomes a zombie (#37011).
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters may raise; must not abort the watcher (CancelledError propagates).
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__), exc_info=True)


# Max seconds between platform reconnect retries (primary watcher and secondary profiles share it).
_RECONNECT_BACKOFF_CAP = 300

# Seconds continuously in the reconnect queue before NEEDS_ATTENTION. Retrying never stops (transient
# outages must self-heal); this only makes a permanently-failing loop loud. 0 disables.
_RECONNECT_ATTENTION_AFTER_SECONDS = _float_env("HERMES_RECONNECT_ATTENTION_AFTER_SECONDS", 7200)


def _reconnect_backoff(attempt: int) -> int:
    """Exponential reconnect backoff: 30s, 60s, 120s, ... capped at 5 min."""
    return min(30 * (2 ** (attempt - 1)), _RECONNECT_BACKOFF_CAP)


def _reconnect_needs_attention(info: dict, now: float) -> bool:
    """True when a reconnect-queue entry has waited long enough for NEEDS_ATTENTION.
    ``queued_at`` is re-stamped on each (re)entry, so only *continuous* failure escalates."""
    if _RECONNECT_ATTENTION_AFTER_SECONDS <= 0:
        return False  # escalation disabled
    queued_at = info.get("queued_at")
    if queued_at is None:
        info["queued_at"] = now
        return False
    return (now - queued_at) >= _RECONNECT_ATTENTION_AFTER_SECONDS


class TurnRunner:
    """Per-turn collaborator carrying the tool-progress callbacks that used to
    be nested closures inside ``GatewayRunner._run_agent_inner``.

    The bodies are byte-identical to the original closures modulo
    ``local_name`` -> ``ctx.field`` rewrites (closed-over locals now travel on
    the shared :class:`gateway.turn_context.TurnContext`) and ``self`` ->
    ``self._runner`` (the owning :class:`GatewayRunner`). Module-global
    references (logger, cfg_get, BasePlatformAdapter, ...) resolve in this
    same module exactly as before.
    """

    def __init__(self, runner: "GatewayRunner", ctx: TurnContext) -> None:
        self._runner = runner
        self._ctx = ctx

    def progress_callback(self, event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
        """Callback invoked by agent on tool lifecycle events."""
        ctx = self._ctx
        # Live status line (Slack's assistant status): stash the current
        # tool phrase on the adapter; the _keep_typing refresh renders it
        # within a couple of seconds. Handled before every other gate
        # because it's independent of progress bubbles and queues (Slack
        # keeps tool_progress off by default, but the ephemeral status
        # line is always safe). Plain dict write — safe from the agent's
        # sync worker thread, no event-loop hop needed.
        if (
            ctx._live_status_adapter is not None
            and ctx._live_status_mode != "off"
            and tool_name != "_thinking"
        ):
            try:
                if event_type == "tool.started" and tool_name and ctx._run_still_current():
                    from agent.display import build_status_phrase
                    _phrase = build_status_phrase(
                        tool_name,
                        args if ctx._live_status_mode == "full" else None,
                    )
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, _phrase)
                elif event_type == "tool.completed":
                    # Between tools the model is genuinely "thinking"
                    # again — revert to the static default.
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, None)
            except Exception as _ls_err:
                logger.debug("live status update failed: %s", _ls_err)
        # "log" mode: append tool.started lines to the log queue and stay
        # silent in chat. Handled before the progress_queue guard because
        # log mode runs without a chat progress queue.
        if ctx.log_queue is not None:
            if event_type == "tool.started" and tool_name and tool_name != "_thinking":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                preview_str = f' "{preview}"' if preview else ""
                ctx.log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
            if not ctx.progress_queue:
                return
        if not ctx.progress_queue or not ctx._run_still_current():
            return

        # First-touch onboarding: the first time a tool takes longer than
        # _LONG_TOOL_THRESHOLD_S during a run that's streaming every tool
        # (progress_mode == "all"), append a one-time hint suggesting
        # /verbose.  We only fire when (a) the user hasn't seen the hint
        # before and (b) /verbose is actually usable on this platform
        # (gateway gate must be open).  The CLI has its own trigger.
        if event_type == "tool.completed" and not ctx.long_tool_hint_fired[0]:
            try:
                duration = kwargs.get("duration") or 0
                if duration >= ctx._LONG_TOOL_THRESHOLD_S and ctx.progress_mode == "all":
                    from agent.onboarding import (
                        TOOL_PROGRESS_FLAG,
                        is_seen,
                        mark_seen,
                        tool_progress_hint_gateway,
                    )
                    _cfg = _load_gateway_config()
                    gate_on = is_truthy_value(
                        cfg_get(_cfg, "display", "tool_progress_command"),
                        default=False,
                    )
                    if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                        ctx.long_tool_hint_fired[0] = True
                        ctx.progress_queue.put(tool_progress_hint_gateway())
                        mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
            except Exception as _hint_err:
                logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
            return

        # "_thinking" is assistant scratch text between tool calls.  It
        # is never ordinary tool progress: only relay it when the platform
        # explicitly opted into thinking_progress.  Handle both legacy
        # callback shapes: ("_thinking", text) and
        # ("reasoning.available", "_thinking", text, ...).
        if event_type == "_thinking" or tool_name == "_thinking":
            if not ctx._thinking_enabled:
                return
            thinking_text = preview if tool_name == "_thinking" else tool_name
            msg = f"💬 {thinking_text}" if thinking_text else None
            if msg:
                ctx.progress_queue.put(msg)
            return

        # Native task cards consume the authoritative ID-bearing
        # tool_start/tool_complete callbacks instead. Do not also enqueue
        # name-correlated text events, which would duplicate cards and
        # mispair concurrent calls to the same tool.
        if ctx._native_slack_task_cards and event_type in {
            "tool.started",
            "tool.completed",
        }:
            return

        # If tool_progress is off, only _thinking passes through (above).
        # Regular tool calls are suppressed.
        if not ctx.tool_progress_enabled:
            return

        # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
        if event_type not in {"tool.started",}:
            return

        # Never render a progress bubble for the clarify tool.  The
        # adapter's send_clarify IS the user-facing rendering (interactive
        # buttons or the numbered-text fallback), so a progress bubble is
        # pure duplication — and in verbose mode it dumps the raw
        # tool-call args JSON ({"question": ..., "choices": [...]}) into
        # the chat.  Because the progress queue drains on a background
        # task, that raw JSON typically lands right underneath the
        # rendered prompt (#52374).
        if tool_name == "clarify":
            return

        # Suppress tool-progress bubbles once the user has sent `stop`.
        # When the LLM response carries N parallel tool calls, the agent
        # fires N "tool.started" events back-to-back before checking for
        # interrupts — without this guard, a late `stop` still renders
        # all N as 🔍 bubbles, making the interrupt feel ignored.
        # (agent lives in run_sync's scope; agent_holder[0] is the shared
        # handle across nested scopes — see line ~9607.)
        try:
            _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent_for_interrupt is not None and getattr(
                _agent_for_interrupt, "is_interrupted", False
            ):
                return
        except Exception:
            pass

        # "new" mode: only report when tool changes
        if ctx.progress_mode == "new" and tool_name == ctx.last_tool[0]:
            return
        ctx.last_tool[0] = tool_name

        # Build progress message with primary argument preview
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")

        # Markdown-capable platforms render a terminal command as a fenced
        # code block instead of the compact `terminal: "cmd…"` preview.
        # Gated on the adapter's ``supports_code_blocks`` capability so
        # plain-text platforms keep the short line.  No language tag is
        # emitted — Slack mrkdwn renders the tag as a literal first code
        # line ("bash"), and a bare fence renders correctly everywhere
        # that supports blocks.
        #
        # Verbose mode shows the FULL command.  Non-verbose ("all"/"new")
        # modes still wrap in a fence but truncate to a single line capped
        # at ``tool_preview_length`` (default 40) so a long or multi-line
        # command doesn't render as a huge block — matching the budget the
        # non-terminal preview path already applies (#42634).
        _code_block_full = None
        _code_block_short = None
        try:
            _progress_adapter = self._runner._adapter_for_source(ctx.source)
        except Exception:
            _progress_adapter = None
        if (
            getattr(_progress_adapter, "supports_code_blocks", False)
            and tool_name == "terminal"
            and isinstance(args, dict)
            and isinstance(args.get("command"), str)
            and args["command"].strip()
        ):
            from agent.display import get_tool_preview_max_len
            _cmd_full = args["command"].rstrip()
            # Consecutive terminal calls: drop the repeated
            # "💻 terminal" header so back-to-back commands render as
            # adjacent code blocks under a single header.
            _block_header = (
                "" if ctx.last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
            )
            _code_block_full = f"{_block_header}```\n{_cmd_full}\n```"
            # Single-line, capped preview for non-verbose modes.
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _lines = _cmd_full.splitlines()
            _cmd_short = _lines[0] if _lines else _cmd_full
            _multiline = len(_lines) > 1
            if len(_cmd_short) > _cap:
                _cmd_short = _cmd_short[:_cap - 3] + "..."
            elif _multiline:
                _cmd_short = _cmd_short + " ..."
            _code_block_short = f"{_block_header}```\n{_cmd_short}\n```"

        # Verbose mode: show detailed arguments, respects tool_preview_length
        if ctx.progress_mode == "verbose":
            if _code_block_full is not None:
                ctx.last_was_terminal_block[0] = True
                ctx.progress_queue.put(_code_block_full)
                return
            ctx.last_was_terminal_block[0] = False
            if args:
                from agent.display import get_tool_preview_max_len
                _pl = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                # When tool_preview_length is 0 (default), don't truncate
                # in verbose mode — the user explicitly asked for full
                # detail.  Platform message-length limits handle the rest.
                if _pl > 0 and len(args_str) > _pl:
                    args_str = args_str[:_pl - 3] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif preview:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            ctx.progress_queue.put(msg)
            return

        # "all" / "new" modes: short preview, respects tool_preview_length
        # config (defaults to 40 chars when unset to keep gateway messages
        # compact — unlike CLI spinners, these persist as permanent messages).
        # Terminal commands on markdown platforms get a single-line capped
        # fenced block (built above) instead of the truncated preview.
        if _code_block_short is not None:
            msg = _code_block_short
            ctx.last_was_terminal_block[0] = True
        elif preview:
            from agent.display import (
                get_tool_preview_max_len,
                get_tool_verb,
                prepare_tool_preview,
                tool_verb_connector,
                verb_drops_preview,
            )
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _prepared_preview = prepare_tool_preview(
                tool_name,
                args,
                fallback=preview,
                max_len=_cap,
            )
            if _progress_adapter is not None:
                preview = _progress_adapter.format_tool_preview(_prepared_preview)
            else:
                preview = _prepared_preview.text
            # Friendly labels: render a human-phrased line for built-in
            # tools ("🔍 Searching the web for ...") by prefixing the verb
            # onto the preview the callback already computed (so the
            # command/url/query is preserved).  Custom/plugin/MCP tools
            # have no verb and fall back to the raw "tool_name: ..." form.
            _verb = get_tool_verb(tool_name)
            if _verb:
                if verb_drops_preview(tool_name):
                    msg = f"{emoji} {_verb}"
                else:
                    msg = f"{emoji} {_verb}{tool_verb_connector(tool_name)}{preview}"
            else:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            ctx.last_was_terminal_block[0] = False
        else:
            msg = f"{emoji} {tool_name}..."
            ctx.last_was_terminal_block[0] = False

        # Dedup: collapse consecutive identical progress messages.
        # Common with execute_code where models iterate with the same
        # code (same boilerplate imports → identical previews).
        if msg == ctx.last_progress_msg[0]:
            ctx.repeat_count[0] += 1
            # Update the last line in progress_lines with a counter
            # via a special "dedup" queue message.
            ctx.progress_queue.put(("__dedup__", msg, ctx.repeat_count[0]))
            return
        ctx.last_progress_msg[0] = msg
        ctx.repeat_count[0] = 0

        ctx.progress_queue.put(msg)

    async def _send_native_task_card_progress(self, adapter) -> None:
        """Drain the progress queue into Slack-native plan/task cards (#29483).

        Consumes the ID-bearing lifecycle dicts queued by
        native_tool_start_callback / native_tool_complete_callback and renders
        them through the adapter's chat.startStream plan/task-card stream.
        On any native failure, falls back to an editable in-thread text
        message so progress stays live for the rest of the turn.
        """
        ctx = self._ctx
        tasks: Dict[str, Dict[str, str]] = {}
        task_order: List[str] = []
        fallback_msg_id: Optional[str] = None
        native_failed = False
        anonymous_seq = 0

        def _compact(value: Any, limit: int = 120) -> str:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3].rstrip() + "..."

        def _visible_tasks() -> List[Dict[str, str]]:
            return [tasks[task_id] for task_id in task_order[-8:]]

        def _fallback_text() -> str:
            labels = {
                "in_progress": "running",
                "complete": "complete",
                "error": "error",
            }
            lines = [
                f"- {task['title']} - {labels.get(task['status'], task['status'])}"
                for task in _visible_tasks()
            ]
            return "Hermes is working\n" + "\n".join(lines)

        def _apply_native_event(raw: Any) -> bool:
            nonlocal anonymous_seq
            if not isinstance(raw, dict):
                return False
            event_type = raw.get("type")
            if event_type not in {"tool.started", "tool.completed"}:
                return False
            call_id = str(raw.get("tool_call_id") or "")
            if not call_id:
                anonymous_seq += 1
                call_id = f"anonymous_{anonymous_seq}"
            tool_name = str(raw.get("tool_name") or "tool")

            if event_type == "tool.started":
                title = tool_name
                preview = _compact(raw.get("preview"), 64)
                if preview:
                    title = f"{tool_name} - {preview}"
                if call_id not in tasks:
                    task_order.append(call_id)
                tasks[call_id] = {
                    "id": call_id,
                    "title": _compact(title),
                    "status": "in_progress",
                }
                return True

            task = tasks.get(call_id)
            if task is None:
                # Completion-only events are rare but valid on some
                # runtimes. Keep their real ID instead of guessing a
                # same-name pending call.
                task = {
                    "id": call_id,
                    "title": _compact(tool_name),
                    "status": "in_progress",
                }
                tasks[call_id] = task
                task_order.append(call_id)
            task["status"] = "error" if raw.get("is_error") else "complete"
            return True

        async def _send_or_edit_fallback() -> None:
            nonlocal fallback_msg_id
            text = _fallback_text()
            if fallback_msg_id:
                result = await adapter.edit_message(
                    chat_id=ctx.source.chat_id,
                    message_id=fallback_msg_id,
                    content=text,
                    metadata=ctx._progress_metadata,
                )
                if getattr(result, "success", False):
                    return
            result = await adapter.send(
                chat_id=ctx.source.chat_id,
                content=text,
                reply_to=ctx._progress_reply_to,
                metadata=ctx._progress_metadata,
            )
            if getattr(result, "success", False) and getattr(
                result, "message_id", None
            ):
                fallback_msg_id = str(result.message_id)
                if ctx._cleanup_progress:
                    ctx._cleanup_msg_ids.append(fallback_msg_id)

        async def _publish_native_progress() -> None:
            nonlocal native_failed
            if not tasks:
                return
            if not native_failed:
                result = await adapter.send_native_task_card_progress(
                    chat_id=ctx.source.chat_id,
                    tasks=_visible_tasks(),
                    title="Hermes is working",
                    reply_to=ctx._progress_reply_to,
                    metadata=ctx._progress_metadata,
                    fallback_text=_fallback_text(),
                )
                if getattr(result, "success", False):
                    return
                native_failed = True
                logger.warning(
                    "Slack native task-card progress failed; falling back "
                    "to an editable text update: %s",
                    getattr(result, "error", "unknown error"),
                )
            # Once the native rail fails, every later lifecycle event
            # edits the same fallback message so progress remains live.
            await _send_or_edit_fallback()

        def _drain_native_queue() -> bool:
            changed = False
            while True:
                try:
                    changed = _apply_native_event(
                        ctx.progress_queue.get_nowait()
                    ) or changed
                except queue.Empty:
                    return changed
                except Exception:
                    logger.debug(
                        "Slack native progress queue drain failed",
                        exc_info=True,
                    )
                    return changed

        def _agent_interrupted() -> bool:
            try:
                _agent = ctx.agent_holder[0] if ctx.agent_holder else None
                return bool(
                    _agent is not None and getattr(_agent, "is_interrupted", False)
                )
            except Exception:
                return False

        try:
            while True:
                if not ctx._run_still_current():
                    return
                try:
                    raw = ctx.progress_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue

                if _agent_interrupted():
                    continue

                if _apply_native_event(raw):
                    await _publish_native_progress()
        except asyncio.CancelledError:
            if _drain_native_queue() and ctx._run_still_current():
                if not _agent_interrupted():
                    await _publish_native_progress()
            return
        finally:
            if hasattr(adapter, "stop_native_task_card_progress"):
                # Best-effort: this finally runs on the turn-cleanup path.
                # An escaping transport exception here propagated through
                # the cleanup awaits (which caught only CancelledError) and
                # skipped final-delivery logic (review B7). Adapters now
                # return failed SendResults, but defend the seam anyway —
                # any adapter, any transport.
                try:
                    await adapter.stop_native_task_card_progress(
                        ctx.source.chat_id,
                        reply_to=ctx._progress_reply_to,
                        metadata=ctx._progress_metadata,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        "task-card stop failed during turn cleanup",
                        exc_info=True,
                    )

    async def send_progress_messages(self):
        ctx = self._ctx
        if not ctx.progress_queue:
            return

        adapter = self._runner._adapter_for_source(ctx.source)
        if not adapter:
            return

        if ctx._native_slack_task_cards and hasattr(
            adapter, "send_native_task_card_progress"
        ):
            await self._send_native_task_card_progress(adapter)
            return

        # Skip tool progress for platforms that don't support message
        # editing (e.g. iMessage/BlueBubbles) — each progress update
        # would become a separate message bubble, which is noisy.
        # getattr, not attribute access: duck-typed adapters (test fakes,
        # minimal plugin adapters) may not define edit_message at all —
        # "missing" means the same thing as "base no-op": can't edit.
        _adapter_edit = getattr(type(adapter), "edit_message", None)
        if _adapter_edit is None or _adapter_edit is BasePlatformAdapter.edit_message:
            while not ctx.progress_queue.empty():
                try:
                    ctx.progress_queue.get_nowait()
                except Exception:
                    break
            return

        progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
        progress_msg_id = None   # ID of the current progress message to edit
        can_edit = ctx.progress_grouping != "separate"  # "separate" = one message per tool (pre-v0.9 behavior)
        _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
        _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

        _progress_len_fn = (
            adapter.message_len_fn
            if isinstance(adapter, BasePlatformAdapter)
            else len
        )
        try:
            _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
        except Exception:
            _raw_progress_limit = 4000
        # Per-chat resolution (relay adapter fronting N platforms): the cap
        # and length unit follow the chat's underlying platform. Native
        # adapters return their scalar/property unchanged.
        if isinstance(adapter, BasePlatformAdapter):
            try:
                _raw_progress_limit = int(
                    adapter.max_message_length_for_chat(ctx.source.chat_id) or 4000
                )
                _progress_len_fn = adapter.message_len_fn_for_chat(ctx.source.chat_id)
            except Exception:
                pass
        # Leave a little room for platform quirks / formatting.  For tiny
        # test adapters keep the limit usable instead of clamping to 500+.
        _PROGRESS_TEXT_LIMIT = max(
            1,
            _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
        )

        # Detect whether the adapter's edit_message accepts metadata so
        # overflow edits preserve Telegram topic/thread routing (#27487).
        _edit_accepts_metadata = False
        if ctx._progress_metadata:
            try:
                _edit_params = inspect.signature(adapter.edit_message).parameters
                _edit_accepts_metadata = (
                    "metadata" in _edit_params
                    or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD
                        for param in _edit_params.values()
                    )
                )
            except (TypeError, ValueError):
                _edit_accepts_metadata = False

        async def _edit_progress_message(message_id: str, content: str):
            kwargs = {
                "chat_id": ctx.source.chat_id,
                "message_id": message_id,
                "content": content,
            }
            if getattr(adapter, "REQUIRES_EDIT_FINALIZE", False):
                kwargs["finalize"] = True
            if _edit_accepts_metadata:
                kwargs["metadata"] = ctx._progress_metadata
            return await adapter.edit_message(**kwargs)

        def _progress_text(lines: list) -> str:
            return "\n".join(str(line) for line in lines)

        def _split_progress_groups(lines: list) -> list[list]:
            """Partition progress lines into platform-sized editable bubbles."""
            groups: list[list] = []
            current: list = []
            for line in lines:
                candidate = current + [line]
                if current and _progress_len_fn(_progress_text(candidate)) > _PROGRESS_TEXT_LIMIT:
                    groups.append(current)
                    current = [line]
                else:
                    current = candidate
            if current:
                groups.append(current)
            return groups

        def _track_progress_result(result) -> None:
            if (
                ctx._cleanup_progress
                and getattr(result, "success", False)
                and getattr(result, "message_id", None)
            ):
                ctx._cleanup_msg_ids.append(str(result.message_id))

        async def _send_progress_text(text: str):
            result = await adapter.send(
                chat_id=ctx.source.chat_id,
                content=text,
                reply_to=ctx._progress_reply_to,
                metadata=ctx._progress_metadata,
            )
            _track_progress_result(result)
            return result

        async def _roll_progress_overflow_if_needed() -> bool:
            """Start fresh editable progress bubbles before a bubble exceeds limit.

                Returns True when it delivered/split the current buffer, or when
                a transient edit failure left the buffer and message identity
                intact for a later retry.  In either case the caller should skip
                the normal send/edit path for this tick.
                """
            nonlocal progress_msg_id, progress_lines, can_edit
            if not progress_lines or not can_edit:
                return False
            groups = _split_progress_groups(progress_lines)
            if len(groups) <= 1:
                return False

            first_text = _progress_text(groups[0])
            if progress_msg_id is not None:
                result = await _edit_progress_message(progress_msg_id, first_text)
                if not result.success:
                    if getattr(result, "retryable", False):
                        logger.debug(
                            "[%s] Transient overflow edit failure — keeping can_edit=True",
                            adapter.name,
                        )
                        return True
                    can_edit = False
                    # Fall back to the existing non-edit behavior below.
                    return False
            else:
                result = await _send_progress_text(first_text)
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            for group in groups[1:]:
                result = await _send_progress_text(_progress_text(group))
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            # The newest continuation is now the only mutable bubble.  Keep
            # just its lines so subsequent edits update it instead of
            # replaying the full historical transcript into new messages.
            progress_lines = groups[-1]
            return True

        while True:
            try:
                if not ctx._run_still_current():
                    while not ctx.progress_queue.empty():
                        try:
                            ctx.progress_queue.get_nowait()
                        except Exception:
                            break
                    return

                raw = ctx.progress_queue.get_nowait()

                # Drain silently when interrupted: events queued in the
                # window between tool parse and interrupt processing
                # should not render as bubbles.  The "⚡ Interrupting
                # current task" message is sent separately and is the
                # last progress-flavored bubble the user should see.
                try:
                    _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
                    if _agent_for_interrupt is not None and getattr(
                        _agent_for_interrupt, "is_interrupted", False
                    ):
                        # Drop this event and continue draining.
                        await asyncio.sleep(0)
                        continue
                except Exception:
                    pass

                # Handle dedup messages: update last line with repeat counter
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if progress_lines:
                        progress_lines[-1] = f"{base_msg} (×{count + 1})"
                    msg = progress_lines[-1] if progress_lines else base_msg
                elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                    # Content bubble just landed on the platform — close off
                    # the current tool-progress bubble so the next tool
                    # starts a fresh bubble below the content. Without this,
                    # tool lines keep editing the ORIGINAL progress message
                    # above the new content, making the chat appear out of
                    # order. Mirrors GatewayStreamConsumer.on_segment_break
                    # on the content side. (Issue: tool + content
                    # linearization regression after PR #7885.)
                    progress_msg_id = None
                    progress_lines = []
                    ctx.last_progress_msg[0] = None
                    ctx.repeat_count[0] = 0
                    continue
                else:
                    msg = raw
                    progress_lines.append(msg)

                if await _roll_progress_overflow_if_needed():
                    _last_edit_ts = time.monotonic()
                    await asyncio.sleep(0.3)
                    if ctx._run_still_current():
                        await adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)
                    continue

                # Throttle edits: batch rapid tool updates into fewer
                # API calls to avoid hitting Telegram flood control.
                # (grammY auto-retry pattern: proactively rate-limit
                # instead of reacting to 429s.)
                _now = time.monotonic()
                _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                if _remaining > 0:
                    # Wait out the throttle interval, then loop back to
                    # drain any additional queued messages before sending
                    # a single batched edit.
                    await asyncio.sleep(_remaining)
                    continue

                if not ctx._run_still_current():
                    return

                if can_edit and progress_msg_id is not None:
                    # Try to edit the existing progress message
                    full_text = "\n".join(progress_lines)
                    result = await _edit_progress_message(progress_msg_id, full_text)
                    if not result.success:
                        _err = (getattr(result, "error", "") or "").lower()
                        # Transient network errors (ConnectError, timeouts)
                        # must not permanently disable progress-message
                        # editing — the next cycle can catch up.  Only
                        # permanent failures (flood control, message not
                        # found, permissions) should set can_edit = False.
                        if getattr(result, "retryable", False):
                            logger.debug(
                                "[%s] Transient edit failure — keeping can_edit=True",
                                adapter.name,
                            )
                            continue
                        if "flood" in _err or "retry after" in _err:
                            # Flood control hit — backoff but keep editing.
                            # Only disable edits for non-recoverable errors.
                            logger.info(
                                "[%s] Progress edit flood control, backing off",
                                adapter.name,
                            )
                            _last_edit_ts = time.monotonic()
                        else:
                            can_edit = False
                        _flood_result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                        if (
                            ctx._cleanup_progress
                            and getattr(_flood_result, "success", False)
                            and getattr(_flood_result, "message_id", None)
                        ):
                            ctx._cleanup_msg_ids.append(str(_flood_result.message_id))
                else:
                    if can_edit:
                        # First tool: send all accumulated text as new message
                        full_text = "\n".join(progress_lines)
                        result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=full_text,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    else:
                        # Editing unsupported: send just this line
                        result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id
                        if ctx._cleanup_progress:
                            ctx._cleanup_msg_ids.append(str(result.message_id))

                _last_edit_ts = time.monotonic()

                # Restore typing indicator
                await asyncio.sleep(0.3)
                if ctx._run_still_current():
                    await adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)

            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                # Drain remaining queued messages
                while not ctx.progress_queue.empty():
                    try:
                        raw = ctx.progress_queue.get_nowait()
                        if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                            _, base_msg, count = raw
                            if progress_lines:
                                progress_lines[-1] = f"{base_msg} (×{count + 1})"
                                await _roll_progress_overflow_if_needed()
                        elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                            # Content-bubble marker during drain: close off
                            # the current progress bubble and start a fresh
                            # one for any tool lines that arrived after.
                            await _roll_progress_overflow_if_needed()
                            if can_edit and progress_lines and progress_msg_id:
                                _pending_text = _progress_text(progress_lines)
                                try:
                                    await _edit_progress_message(progress_msg_id, _pending_text)
                                except Exception:
                                    pass
                            progress_msg_id = None
                            progress_lines = []
                            ctx.last_progress_msg[0] = None
                            ctx.repeat_count[0] = 0
                        else:
                            progress_lines.append(raw)
                            await _roll_progress_overflow_if_needed()
                    except Exception:
                        break
                # Final edit with all remaining tools (only if editing works)
                if can_edit and progress_lines and progress_msg_id:
                    await _roll_progress_overflow_if_needed()
                if can_edit and progress_lines and progress_msg_id:
                    full_text = _progress_text(progress_lines)
                    try:
                        await _edit_progress_message(progress_msg_id, full_text)
                    except Exception:
                        pass
                return
            except Exception as e:
                logger.error("Progress message error: %s", e)
                await asyncio.sleep(1)

    def voice_ack_callback(self, call_id, tool_name, args):
        """tool_start_callback: speak a one-time ack in the voice channel."""
        ctx = self._ctx
        if ctx._voice_ack_fired[0] or ctx._voice_ack_guild[0] is None:
            return
        if not ctx._run_still_current():
            return
        ctx._voice_ack_fired[0] = True
        _adapter = self._runner.adapters.get(Platform.DISCORD)
        if _adapter is None or not hasattr(_adapter, "play_ack_in_voice"):
            return
        try:
            safe_schedule_threadsafe(
                _adapter.play_ack_in_voice(ctx._voice_ack_guild[0]),
                ctx._voice_ack_loop,
                logger=logger,
                log_message="voice ack scheduling error",
            )
        except Exception as _ack_err:
            logger.debug("voice ack schedule failed: %s", _ack_err)

    # ── Slack-native task cards: ID-bearing lifecycle callbacks (#29483) ──
    # These ride agent.tool_start_callback / agent.tool_complete_callback so
    # start/completion events correlate by the REAL tool-call id — the
    # name-correlated text events in progress_callback would duplicate cards
    # and mispair concurrent calls to the same tool.

    def native_tool_start_callback(self, call_id, tool_name, args):
        """Queue an ID-correlated native progress start from the agent thread."""
        ctx = self._ctx
        if not ctx.progress_queue or not ctx._run_still_current():
            return
        try:
            _agent = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent is not None and getattr(_agent, "is_interrupted", False):
                return
        except Exception:
            pass
        from agent.display import build_tool_preview

        ctx.progress_queue.put(
            {
                "type": "tool.started",
                "tool_call_id": str(call_id or ""),
                "tool_name": str(tool_name or "tool"),
                "preview": build_tool_preview(
                    str(tool_name or "tool"), args or {}, max_len=64
                )
                or "",
            }
        )

    def native_tool_complete_callback(self, call_id, tool_name, args, result):
        """Queue the matching native completion using the real tool-call ID."""
        ctx = self._ctx
        if not ctx.progress_queue or not ctx._run_still_current():
            return
        try:
            _agent = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent is not None and getattr(_agent, "is_interrupted", False):
                return
        except Exception:
            pass
        from agent.display import _detect_tool_failure

        is_error, _ = _detect_tool_failure(str(tool_name or "tool"), result)
        ctx.progress_queue.put(
            {
                "type": "tool.completed",
                "tool_call_id": str(call_id or ""),
                "tool_name": str(tool_name or "tool"),
                "is_error": bool(is_error),
            }
        )

    def combined_tool_start_callback(self, call_id, tool_name, args):
        """Compose the voice ack + native task-card start consumers."""
        ctx = self._ctx
        if ctx._voice_ack_guild[0] is not None:
            self.voice_ack_callback(call_id, tool_name, args)
        if ctx._native_slack_task_cards:
            self.native_tool_start_callback(call_id, tool_name, args)

    def _step_callback_sync(self, iteration: int, prev_tools: list) -> None:
        ctx = self._ctx
        if not ctx._run_still_current():
            return
        # prev_tools may be list[str] or list[dict] with "name"/"result"
        # keys.  Normalise to keep "tool_names" backward-compatible for
        # user-authored hooks that do ', '.join(tool_names)'.
        _names: list[str] = []
        for _t in (prev_tools or []):
            if isinstance(_t, dict):
                _names.append(_t.get("name") or "")
            else:
                _names.append(str(_t))
        safe_schedule_threadsafe(
            ctx._hooks_ref.emit("agent:step", {
                "platform": ctx.source.platform.value if ctx.source.platform else "",
                "user_id": ctx.source.user_id,
                "session_id": ctx.session_id,
                "iteration": iteration,
                "tool_names": _names,
                "tools": prev_tools,
            }),
            ctx._loop_for_step,
            logger=logger,
            log_message="agent:step hook scheduling error",
        )

    def _event_callback_sync(self, event_type: str, context: dict) -> None:
        ctx = self._ctx
        try:
            asyncio.run_coroutine_threadsafe(
                ctx._hooks_ref.emit(event_type, context),
                ctx._loop_for_step,
            )
        except Exception as _e:
            logger.debug("event_callback hook error: %s", _e)

    def _attach_session_title_callback(self, agent, ctx) -> None:
        """Wire the platform thread-rename lane onto the agent as `_on_session_title`.

        The session titler runs inside the turn prologue now (it derives the
        title from the user's first message, so it no longer needs the
        response), which means the callback has to be attached before the run
        rather than registered after it. The lane predicates and their
        rationale are unchanged from the old post-response registration.
        """
        try:
            # Gateway auto-title failures must NOT be surfaced as user-visible
            # messages (#23246) — they are not actionable to the end user.
            # Overriding the failure sink here keeps CLI mode on the agent's
            # _emit_auxiliary_failure path while the gateway logs at debug.
            def _title_failure_cb(task: str, exc: BaseException) -> None:
                logger.debug(
                    "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                    task, exc,
                )

            agent._title_failure_callback = _title_failure_cb

            session_id = getattr(agent, "session_id", None)
            source = ctx.source

            # Both lanes below spend a rate-limited platform call per title, so
            # they take the model's title and skip the derived one — see
            # TitleCallback. Renaming twice lands on the same name at twice the
            # cost, and Discord's 2-per-10-minutes channel budget can spend
            # itself on the throwaway and drop the one worth showing.
            if self._runner._is_telegram_topic_lane(source):
                agent._on_session_title = lambda title, title_source: (
                    title_source == "llm"
                    and self._runner._schedule_telegram_topic_title_rename(
                        source, session_id, title,
                    )
                )
            elif self._runner._is_discord_auto_thread_lane(source) or (
                self._runner._is_relay_discord_channel_lane(source)
            ):
                # Relay note: the second predicate is shape-only (relay
                # Discord channel event). Whether the connector actually
                # auto-threaded our reply is only knowable AFTER delivery
                # (send-result feedback), so the callback must be registered
                # eagerly and the rename lane performs the cache lookup at
                # fire time (staging repro 2026-07-31: gating registration on
                # the cache read meant it never registered and no
                # thread_rename op was ever sent).
                agent._on_session_title = lambda title, title_source: (
                    title_source == "llm"
                    and self._runner._schedule_discord_semantic_thread_rename(
                        source, session_id, title,
                    )
                )
        except Exception:
            logger.debug("Failed to attach session title callback", exc_info=True)

    def _status_callback_sync(self, event_type: str, message: str) -> None:
        ctx = self._ctx
        if not ctx._status_adapter or not ctx._run_still_current():
            return
        prepared_message = _prepare_gateway_status_message(
            ctx.source.platform,
            event_type,
            message,
        )
        if prepared_message is None:
            logger.debug(
                "status_callback suppressed for %s/%s: %s",
                ctx.source.platform.value if ctx.source.platform else "unknown",
                event_type,
                _redact_gateway_user_facing_secrets(str(message or ""))[:160],
            )
            return
        _fut = safe_schedule_threadsafe(
            _send_or_update_status_coro(ctx._status_adapter, ctx._status_chat_id, event_type, prepared_message, ctx._status_thread_metadata),
            ctx._loop_for_step,
            logger=logger,
            log_message=f"status_callback ({event_type}) scheduling error",
        )
        if _fut is None:
            return
        if ctx._cleanup_progress:
            def _track_status_id(fut) -> None:
                try:
                    res = fut.result()
                except Exception:
                    return
                mid = getattr(res, "message_id", None)
                if getattr(res, "success", False) and mid:
                    ctx._cleanup_msg_ids.append(str(mid))
            _fut.add_done_callback(_track_status_id)

    def run_sync(self):
        ctx = self._ctx
        # Historical note: as a nested closure this body declared
        # `nonlocal message` because the conditional re-assignments below
        # (prepending model-switch / resume-recovery notes) would otherwise
        # make `message` function-local and break the earlier read at
        # `_resolve_turn_agent_config(message, …)`.  As a method the turn
        # message lives on the shared TurnContext instead: every rebind
        # writes `ctx.message`, so the outer `_run_agent_inner` body observes
        # the updated value exactly as it did through the closure cell.

        # session_key is propagated via contextvars in _set_session_env()
        # (_SESSION_KEY) and via set_current_session_key() (_approval_session_key)
        # below — both concurrency-safe and inherited by tool worker threads.
        # We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
        # os.environ is process-global, so concurrent gateway sessions (e.g.
        # two Discord threads) would clobber each other's value, and a tool
        # thread whose contextvar is unset would fall back to os.environ and
        # read the wrong session key — misrouting command-approval prompts to
        # the wrong thread (#24100). The non-gateway surfaces don't depend on
        # this write: CLI and cron bind the session via contextvars
        # (set_current_session_key / session context), and only the TUI
        # slash-worker *subprocess* exports HERMES_SESSION_KEY (from its own
        # --session-key argv, a separate process) — so removing this in-process
        # gateway write does not affect any of them.

        # Map platform enum to the platform hint key the agent understands.
        # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
        platform_key = "cli" if ctx.source.platform == Platform.LOCAL else ctx.source.platform.value

        # Combine platform context, YAML channel_prompts hint for this chat,
        # channel_overrides system_prompt (or global ephemeral), and gateway
        # ephemeral prompt from _get_system_prompt_for_channel.
        combined_ephemeral = ctx.context_prompt or ""
        event_channel_prompt = (ctx.channel_prompt or "").strip()
        if event_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
        cfg_channel_prompt = self._runner._get_system_prompt_for_channel(
            ctx.source.platform,
            ctx.source.chat_id or "",
            thread_id=getattr(ctx.source, "thread_id", None),
            parent_id=getattr(ctx.source, "parent_chat_id", None),
        )
        if cfg_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + cfg_channel_prompt).strip()

        voice_fast_lane = bool(getattr(ctx.source, "_voice_fast_lane", False))
        voice_fast_lane_task = (
            voice_fast_lane
            and self._runner._voice_fast_lane_requests_work(ctx.message or "")
        )
        if voice_fast_lane:
            if voice_fast_lane_task:
                lane_prompt = (
                    "[Voice fast lane: the user explicitly requested work. "
                    "Proceed normally, but acknowledge the task briefly before details.]"
                )
            else:
                lane_prompt = (
                    "[Voice fast lane: answer in one to three brief, natural spoken "
                    "sentences. Do not offer or attempt tools, background work, or a "
                    "multi-step plan unless the user explicitly asks you to perform work.]"
                )
            combined_ephemeral = (combined_ephemeral + "\n\n" + lane_prompt).strip()

        max_iterations = _current_max_iterations()
        if voice_fast_lane and not voice_fast_lane_task:
            max_iterations = min(max_iterations, 2)

        try:
            model, runtime_kwargs = self._runner._resolve_session_agent_runtime(
                source=ctx.source,
                session_key=ctx.session_key,
                user_config=ctx.user_config,
            )
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), ctx.session_key or "",
            )
        except Exception as exc:
            return {
                "final_response": f"⚠️ Provider authentication failed: {exc}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        pr = self._runner._provider_routing
        reasoning_config = self._runner._resolve_session_reasoning_config(
            source=ctx.source,
            session_key=ctx.session_key,
            model=model,
        )
        self._runner._reasoning_config = reasoning_config
        self._runner._service_tier = self._runner._resolve_session_service_tier(
            source=ctx.source, session_key=ctx.session_key
        )
        # Set up stream consumer for token streaming or interim commentary.
        _stream_consumer = None
        _stream_delta_cb = None
        # #60671 — streaming TTS consumer is created on the outer
        # event-loop thread before run_sync launches.  run_sync only
        # reads it via ``streaming_tts_consumer_holder[0]`` for delta
        # callback wiring.
        _stts_consumer_ref = ctx.streaming_tts_consumer_holder[0]
        _scfg = getattr(getattr(self._runner, 'config', None), 'streaming', None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        # Per-platform streaming gate: display.platforms.<plat>.streaming
        # can disable streaming for specific platforms even when the global
        # streaming config is enabled.
        _plat_streaming = ctx.resolve_display_setting(
            ctx.user_config, platform_key, "streaming"
        )
        # None = no per-platform override → follow global config
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )
        _want_stream_deltas = _streaming_enabled
        _want_interim_messages = ctx.interim_assistant_messages_enabled
        _want_interim_consumer = _want_interim_messages
        if _want_stream_deltas or _want_interim_consumer:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                _adapter = self._runner._adapter_for_source(ctx.source)
                if _adapter:
                    _consumer_cfg, _pause_typing_before_finalize = (
                        self._runner._build_stream_consumer_config(
                            ctx.source, _scfg, _adapter,
                            on_missing_cursor="raise",
                        )
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=ctx.source.chat_id,
                        config=_consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",)))
                            if ctx.progress_queue is not None
                            else None
                        ),
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id,
                        run_still_current=ctx._run_still_current,
                    )
                    if _want_stream_deltas:
                        def _stream_delta_cb(text: str) -> None:
                            if ctx._run_still_current():
                                _stream_consumer.on_delta(text)
                                # Tee to the streaming-TTS consumer (#60671).
                                if _stts_consumer_ref is not None:
                                    _stts_consumer_ref.on_delta(text)
                    ctx.stream_consumer_holder[0] = _stream_consumer
            except Exception as _sc_err:
                logger.debug("Could not set up stream consumer: %s", _sc_err)

        # When text streaming is off but streaming TTS is active,
        # install a TTS-only delta callback so the consumer still
        # receives LLM deltas for audio synthesis (#60671).
        if _stream_delta_cb is None and _stts_consumer_ref is not None:
            def _stream_delta_cb(text: str) -> None:
                if ctx._run_still_current():
                    _stts_consumer_ref.on_delta(text)

        def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            display_text = text
            if _stream_consumer is not None:
                if already_streamed:
                    _stream_consumer.on_segment_break()
                else:
                    _stream_consumer.on_commentary(display_text)
                return
            if already_streamed or not ctx._status_adapter or not str(display_text or "").strip():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    display_text,
                    metadata=ctx._status_thread_metadata,
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="interim_assistant_callback scheduling error",
            )

        turn_route = self._runner._resolve_turn_agent_config(ctx.message, model, runtime_kwargs)

        # Per-platform skip_context_files — messaging platforms can opt out
        # of filesystem-heavy context-file discovery (SOUL.md, AGENTS.md,
        # .cursorrules) to cut AIAgent construction latency. Especially
        # impactful on Windows, where stat() + directory walks are 10-100x
        # slower than Linux. Off by default; soul identity is preserved so
        # the persona survives even with minimal context.
        _platforms_gw_cfg = (ctx.user_config.get("gateway") or {}).get("platforms") or {}
        # ``hermes gateway setup`` writes ``gateway.platforms`` as a LIST of
        # enabled platform names (e.g. ``- telegram``), not a dict.  Treat any
        # non-dict shape as "no per-platform overrides" instead of crashing
        # on ``.get()`` for every incoming turn (#83185).
        if not isinstance(_platforms_gw_cfg, dict):
            _platforms_gw_cfg = {}
        _plat_gw_cfg = _platforms_gw_cfg.get(platform_key) or {}
        _skip_context = _plat_gw_cfg.get("skip_context_files")
        skip_context_files = bool(_skip_context) if _skip_context is not None else False
        if voice_fast_lane and not voice_fast_lane_task:
            # The persona remains loaded, while expensive project-context
            # discovery stays out of short spoken exchanges.
            skip_context_files = True

        # Check agent cache — reuse the AIAgent from the previous message
        # in this session to preserve the frozen system prompt and tool
        # schemas for prompt cache hits.
        _sig = self._runner._agent_config_signature(
            turn_route["model"],
            turn_route["runtime"],
            ctx.enabled_toolsets,
            combined_ephemeral,
            cache_keys=self._runner._extract_cache_busting_config(ctx.user_config),
            user_id=getattr(ctx.source, "user_id", None),
            user_id_alt=getattr(ctx.source, "user_id_alt", None),
            skip_context_files=skip_context_files,
        )
        agent = None
        reused_cached_agent = False
        _cache_lock = getattr(self._runner, "_agent_cache_lock", None)
        _cache = getattr(self._runner, "_agent_cache", None)

        # Peek at the cached entry's snapshot session_id (if any) so we can
        # check, OUTSIDE the cache lock, whether THAT session_id is a DEAD
        # session in state.db. This closes a gap in the #54947 fix: that
        # fix treats "cached session_id != current session_id" as an
        # intentional /resume-style switch and reuses the agent unchanged.
        # But the #54878 self-heal produces the exact same tuple shape
        # when it recovers a routing key away from a session that was
        # already ended — the cached AIAgent still belongs to the DEAD
        # session, not a valid sibling conversation. Reusing it lets that
        # turn's post-run "session split" sync write the routing key
        # straight back onto the dead session_id, undoing the self-heal
        # and looping every message until an interrupt happens to race in
        # first (the #54878 x #54947 interaction — no existing upstream
        # issue tracks this combination as of 2026-07-12).
        _peek_cached_sid = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                _peek_entry = _cache.get(ctx.session_key)
            if _peek_entry and len(_peek_entry) > 3:
                _peek_cached_sid = _peek_entry[3]
        _cached_sid_is_dead = False
        if (
            _peek_cached_sid is not None
            and ctx.session_id is not None
            and _peek_cached_sid != ctx.session_id
        ):
            try:
                _cached_sid_is_dead = self._runner.session_store._is_session_ended_in_db(
                    _peek_cached_sid
                )
            except Exception:
                _cached_sid_is_dead = False

        # Detect cross-process writes: when another process (e.g. hermes
        # dashboard) appends to the same session in the shared SessionDB,
        # the cached agent's in-memory transcript becomes stale.  Compare
        # the session's current message_count against the count recorded
        # when the agent was cached; on mismatch, invalidate the cache
        # so a fresh agent re-reads from disk. (#45966)
        _current_msg_count = None
        if self._runner._session_db is not None and ctx.session_id:
            try:
                # run_sync is off-loop (executor); sync DB is fine.
                _sess_row = self._runner._session_db._db.get_session(ctx.session_id)
                if _sess_row:
                    _current_msg_count = _sess_row.get("message_count", 0)
            except Exception:
                pass

        _xproc_evicted_agent = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                cached = _cache.get(ctx.session_key)
                if cached and cached[1] == _sig:
                    # cached[2] is the message_count at cache time;
                    # stale when a second process appended rows.
                    # cached[3] (when present) is the session_id the
                    # snapshot was taken for — used to skip the guard
                    # when the active session_id differs (#54947).
                    _cached_mc = cached[2] if len(cached) > 2 else None
                    _cached_sid = cached[3] if len(cached) > 3 else None
                    # If the snapshot belongs to a different session_id
                    # (same session_key, different conversation), the
                    # message_count comparison is meaningless — the
                    # counts track DIFFERENT DB rows.  REUSE the cached
                    # agent rather than rebuild and bust the prompt cache
                    # on every session switch (#54947).
                    _session_id_mismatch = (
                        _cached_sid is not None
                        and ctx.session_id is not None
                        and _cached_sid != ctx.session_id
                    )
                    # Re-validate the OUTSIDE-lock dead-session peek
                    # against the tuple actually read under THIS lock —
                    # the cache entry could have been replaced between
                    # the peek and this lock acquisition, and a stale
                    # "dead" verdict must never be applied to a
                    # different (possibly live) cached agent.
                    _stale_dead_sid_reuse = (
                        _session_id_mismatch
                        and _cached_sid_is_dead
                        and _cached_sid == _peek_cached_sid
                    )
                    if _stale_dead_sid_reuse:
                        # #54878 x #54947 interaction: the routing key
                        # was just self-healed away from a session that
                        # state.db already marked ended, but the cached
                        # AIAgent here still belongs to that DEAD
                        # session_id. The #54947 "different session_id
                        # under the same key = intentional switch, reuse
                        # freely" rule does not hold here — this isn't a
                        # sibling conversation, it's a stale agent left
                        # over from before the self-heal. Reusing it lets
                        # this turn's post-run "session split" sync write
                        # the routing key straight back onto the dead
                        # session_id, undoing the self-heal and looping
                        # every message until an interrupt happens to
                        # race in first. Discard and rebuild fresh
                        # instead, same as a genuine cross-process write.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "cached agent's session_id %s is ended in "
                            "state.db (stale self-heal artifact, "
                            "#54878 x #54947) — discarding instead of "
                            "reusing across the routing recovery",
                            ctx.session_key, _cached_sid,
                        )
                        evicted = self._runner._agent_cache.pop(ctx.session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Same deferred-cleanup rationale as the
                            # cross-process branch below (#52197): don't
                            # block the event loop / cache lock on
                            # memory-provider shutdown or socket teardown.
                            _xproc_evicted_agent = _ev_agent
                    elif (
                        not _session_id_mismatch
                        and _cached_mc is not None
                        and _current_msg_count is not None
                        and _current_msg_count != _cached_mc
                    ):
                        # Cross-process write detected — discard stale
                        # agent so it rebuilds from fresh DB transcript.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "message_count changed (%s -> %s), "
                            "possible cross-process write",
                            ctx.session_key, _cached_mc, _current_msg_count,
                        )
                        evicted = self._runner._agent_cache.pop(ctx.session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Defer cleanup until AFTER the lock is
                            # released — _cleanup_agent_resources /
                            # release_clients can block on memory-provider
                            # shutdown and socket teardown, and running it
                            # here would stall the gateway event loop while
                            # _sweep_idle_cached_agents (session-expiry
                            # watcher) waits on the same lock, blocking
                            # Discord heartbeats (#52197).  The same session
                            # rebuilds a fresh agent immediately below, so
                            # use the SOFT release that preserves the
                            # session's terminal sandbox / browser / bg
                            # processes for the rebuilt agent to inherit —
                            # mirrors _evict_cached_agent / idle-sweep.
                            _xproc_evicted_agent = _ev_agent
                    else:
                        agent = cached[0]
                        # Refresh LRU order so the cap enforcement evicts
                        # truly-oldest entries, not the one we just used.
                        if hasattr(_cache, "move_to_end"):
                            try:
                                _cache.move_to_end(ctx.session_key)
                            except KeyError:
                                pass
                        self._runner._init_cached_agent_for_turn(agent, ctx._interrupt_depth)
                        # Refresh agent max_iterations from current config
                        # (cached agent may have been created with old config)
                        agent.max_iterations = max_iterations
                        logger.debug("Reusing cached agent for session %s", ctx.session_key)
                        reused_cached_agent = True

        # Lock released — refresh the fallback chain from disk for the
        # reused agent OUTSIDE the cache lock (config.yaml read is disk
        # I/O; the idle-sweep watcher contends on this lock and stalls
        # Discord heartbeats — same reasoning as #52197).  A chain
        # configured after this agent was cached (or after gateway start)
        # must reach the next turn (#60955).  Per-session turn
        # serialization (_running_agents) keeps this safe post-lock.
        if reused_cached_agent and agent is not None:
            self._runner._apply_fallback_chain_to_agent(
                agent, self._runner._refresh_fallback_model(),
            )

        # Lock released — now schedule cleanup of any cross-process-evicted
        # agent on a daemon thread so memory-provider shutdown / socket
        # teardown never blocks the gateway event loop or the cache lock
        # the session-expiry watcher needs (#52197).
        if _xproc_evicted_agent is not None:
            try:
                threading.Thread(
                    target=self._runner._release_evicted_agent_soft,
                    args=(_xproc_evicted_agent,),
                    daemon=True,
                    name=f"agent-xproc-evict-{str(ctx.session_key)[:24]}",
                ).start()
            except Exception:
                # Interpreter shutdown or thread-spawn failure — release
                # inline as a best-effort fallback.
                try:
                    self._runner._release_evicted_agent_soft(_xproc_evicted_agent)
                except Exception:
                    pass

        if agent is None:
            # Config changed or first message — create fresh agent
            agent = ctx.AIAgent(
                model=turn_route["model"],
                **turn_route["runtime"],
                **_checkpoint_agent_kwargs(ctx.user_config),
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=ctx.enabled_toolsets,
                disabled_toolsets=ctx.disabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._runner._prefill_messages or None,
                reasoning_config=reasoning_config,
                service_tier=self._runner._service_tier,
                request_overrides=turn_route.get("request_overrides"),
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=ctx.session_id,
                platform=platform_key,
                user_id=ctx.source.user_id,
                user_id_alt=ctx.source.user_id_alt,
                user_name=ctx.source.user_name,
                chat_id=ctx.source.chat_id,
                chat_name=ctx.source.chat_name,
                chat_type=ctx.source.chat_type,
                thread_id=ctx.source.thread_id,
                gateway_session_key=ctx.session_key,
                session_db=getattr(self._runner._session_db, "_db", self._runner._session_db),
                # Reload from disk — do not reuse the startup snapshot (#60955).
                fallback_model=self._runner._refresh_fallback_model(),
                skip_context_files=skip_context_files,
                # Keep the persona even with minimal context: soul identity is
                # a single small file, not part of the expensive walk.
                load_soul_identity=True,
            )
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    # Record the session_id the snapshot was taken for
                    # alongside the message_count, so the cross-process
                    # guard can skip the (meaningless) count comparison
                    # when the active session_id later switches under
                    # the same session_key (#54947).
                    _cache[ctx.session_key] = (
                        agent, _sig, _current_msg_count, ctx.session_id,
                    )
                    self._runner._enforce_agent_cache_cap()
            logger.debug("Created new agent for session %s (sig=%s)", ctx.session_key, _sig)

        # Per-message state — callbacks and reasoning config change every
        # turn and must not be baked into the cached agent constructor.
        # Gate on needs_progress_queue (tool_progress OR thinking_progress)
        # rather than tool_progress alone: the progress_callback also relays
        # _thinking assistant scratch text, which is gated on
        # thinking_progress and is intentionally independent of tool
        # progress. With the old `tool_progress_enabled`-only gate, a user
        # who set thinking_progress:true but kept tool_progress:off got a
        # None callback — so _thinking scratch bubbles never relayed even
        # though the progress queue was created for them.
        agent.tool_progress_callback = (
            ctx.progress_callback
            if (
                ctx.needs_progress_queue
                or ctx.log_mode_enabled
                or ctx._live_status_adapter is not None
            )
            else None
        )
        # Compose ID-bearing lifecycle consumers: Discord's one-time voice
        # ack and Slack's native task cards both ride the authoritative
        # start callback, so neither has to infer identity from tool names.
        _combined_start_cb = ctx.native_tool_start_callback or ctx.voice_ack_callback
        agent.tool_start_callback = (
            _combined_start_cb
            if (
                ctx._voice_ack_guild[0] is not None
                or ctx._native_slack_task_cards
            )
            else None
        )
        agent.tool_complete_callback = (
            ctx.native_tool_complete_callback
            if ctx._native_slack_task_cards
            and ctx.native_tool_complete_callback is not None
            else None
        )
        agent.step_callback = ctx._step_callback_sync if ctx._hooks_ref.loaded_hooks else None
        agent.stream_delta_callback = _stream_delta_cb
        agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
        agent.status_callback = ctx._status_callback_sync
        # Credits / out-of-band notices (usage bands, depletion, restored).
        # Messaging has no persistent status bar, so each notice is a
        # standalone push: render to a single plaintext line and deliver via
        # the shared _deliver_platform_notice rail (honors private/public +
        # thread metadata). Fires from the agent's sync worker thread, so we
        # hop onto the gateway loop with safe_schedule_threadsafe - same
        # pattern as _status_callback_sync. The fired-once latch lives on the
        # cached agent and persists across turns, so a band crosses -> one
        # push (no per-turn re-nag). Recovery ("✓ Credit access restored")
        # rides the same show path (it's emitted as a success notice, not a
        # clear). The clear callback is a no-op: a sent platform message
        # can't be cleanly retracted, and the band already fired once.
        def _notice_callback_sync(notice) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            try:
                line = render_notice_line(notice)
            except Exception:
                logger.debug("render_notice_line failed", exc_info=True)
                return
            if not line:
                return
            safe_schedule_threadsafe(
                self._runner._deliver_platform_notice(ctx.source, line),
                ctx._loop_for_step,
                logger=logger,
                log_message="notice_callback delivery scheduling error",
            )

        agent.notice_callback = _notice_callback_sync
        agent.notice_clear_callback = None
        agent.event_callback = ctx._event_callback_sync
        agent.reasoning_config = reasoning_config
        agent.service_tier = self._runner._service_tier
        agent.request_overrides = turn_route.get("request_overrides") or {}
        # Must-deliver notes for THIS turn ride the current user message
        # (api_content sidecar), never the system prompt: staged by
        # _handle_message_with_agent (auto-reset note, first-contact
        # intro, voice-channel change).  Assigned unconditionally so a
        # reused cached agent never replays a stale note.
        agent._gateway_turn_context_notes = "\n\n".join(
            self._runner._consume_pending_turn_sidecar_notes(ctx.session_key)
        )

        _bg_review_release = threading.Event()
        _bg_review_pending: list[str] = []
        _bg_review_pending_lock = threading.Lock()

        def _deliver_bg_review_message(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    message,
                    metadata=_interim_metadata(_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform)),
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="background_review_callback scheduling error",
            )

        def _release_bg_review_messages() -> None:
            _bg_review_release.set()
            with _bg_review_pending_lock:
                pending = list(_bg_review_pending)
                _bg_review_pending.clear()
            for queued in pending:
                _deliver_bg_review_message(queued)

        # Background review delivery — send "💾 Memory updated" etc. to user
        def _bg_review_send(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            if not _bg_review_release.is_set():
                with _bg_review_pending_lock:
                    if not _bg_review_release.is_set():
                        _bg_review_pending.append(message)
                        return
            _deliver_bg_review_message(message)

        agent.background_review_callback = _bg_review_send
        # Register the release hook on the adapter so base.py's finally
        # block can fire it after delivering the main response.
        if ctx._status_adapter and ctx.session_key:
            if getattr(type(ctx._status_adapter), "register_post_delivery_callback", None) is not None:
                ctx._status_adapter.register_post_delivery_callback(
                    ctx.session_key,
                    _release_bg_review_messages,
                    generation=ctx.run_generation,
                )
            else:
                _pdc = getattr(ctx._status_adapter, "_post_delivery_callbacks", None)
                if _pdc is not None:
                    _pdc[ctx.session_key] = _release_bg_review_messages
        # Memory update notifications in chat.  Config: display.memory_notifications
        #   off     — no chat notification (still logged to stdout)
        #   on      — generic "💾 Memory updated" (default)
        #   verbose — content preview: "💾 Memory ➕ Hermes Repo..."
        _mem_notif = ctx.user_config.get("display", {}).get("memory_notifications")
        if isinstance(_mem_notif, bool):
            _mem_notif = "on" if _mem_notif else "off"
        agent.memory_notifications = str(_mem_notif).lower() if _mem_notif else "on"

        # ------------------------------------------------------------------
        # Clarify callback: present a clarify prompt and block on a response.
        #
        # Runs on the agent's worker thread (see clarify_tool's synchronous
        # callback contract).  Bridges sync→async by scheduling the
        # adapter's send_clarify on the gateway event loop, then blocks on
        # the clarify primitive's threading.Event with a configurable
        # timeout.  Returns the user's response string, or a sentinel
        # explaining that no response arrived (so the agent can adapt
        # rather than hang forever).
        # ------------------------------------------------------------------
        def _clarify_callback_sync(question: str, choices, multi_select: bool = False) -> str:
            from tools import clarify_gateway as _clarify_mod
            import uuid as _uuid

            if not ctx._status_adapter:
                return ""

            clarify_id = _uuid.uuid4().hex[:10]
            _clarify_mod.register(
                clarify_id=clarify_id,
                session_key=ctx.session_key or "",
                question=question,
                choices=list(choices) if choices else None,
                multi_select=bool(multi_select),
            )

            # Pause typing — like approval, we don't want a "thinking..."
            # status to obscure the prompt or block the user from typing
            # an "Other" response on platforms that disable input while
            # typing is active (Slack Assistant API).
            try:
                ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)
            except Exception:
                pass

            # Ordering barrier (#clarify-ordering): flush any buffered
            # assistant prose (interim commentary / streamed deltas) to the
            # platform BEFORE sending the poll.  The poll is delivered on a
            # separate, agent-thread-blocking path; without this barrier it
            # races ahead of prose still sitting in the stream consumer's
            # queue, so the question renders ABOVE its own explanation.
            # Best-effort + short timeout: never hang the agent thread if
            # the consumer task isn't running.
            try:
                _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
                _flush = getattr(_sc, "flush_pending_sync", None)
                if callable(_flush):
                    _flush(timeout=3.0)
            except Exception:
                logger.debug(
                    "Stream-consumer flush before clarify prompt failed",
                    exc_info=True,
                )

            fut = safe_schedule_threadsafe(
                ctx._status_adapter.send_clarify(
                    chat_id=ctx._status_chat_id,
                    question=question,
                    choices=list(choices) if choices else None,
                    clarify_id=clarify_id,
                    session_key=ctx.session_key or "",
                    metadata=ctx._status_thread_metadata,
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="Clarify send failed to schedule",
            )
            # Boundary rule (see _approval_send_outcome): a send timeout is
            # AMBIGUOUS — the card may have posted with a late ack. Only a
            # definitive failure tears down the registration; ambiguous
            # falls through to the bounded wait so a late reply resolves.
            return _clarify_send_then_wait(
                fut,
                clarify_id=clarify_id,
                session_key=ctx.session_key or "",
                clarify_mod=_clarify_mod,
            )

        agent.clarify_callback = _clarify_callback_sync

        # Show assistant thinking between tool calls — independent of
        # tool_progress mode. Mattermost needs an explicit per-platform
        # opt-in so global scratch-text display does not leak into threads.
        agent.thinking_progress = ctx._thinking_enabled
        # Store agent reference for interrupt support
        ctx.agent_holder[0] = agent
        # Wire the platform thread-rename lane onto the agent, because the
        # session titler now fires from the turn prologue rather than after
        # the response. Titles are pushed here the moment they land.
        self._attach_session_title_callback(agent, ctx)
        # Publish turn ownership for explicit /stop, /new, disconnect, and
        # shutdown interrupts. Older session processes are outside this
        # baseline and remain alive.
        agent._gateway_turn_process_task_id = ctx.process_task_id
        agent._gateway_turn_process_baseline = ctx.process_baseline
        # Capture the full tool definitions for transcript logging
        ctx.tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None

        # Convert history to agent format.
        # Two cases:
        #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
        #      - Strip timestamps, keep role+content
        #   2. Interrupt path (from agent result["messages"]): full agent messages
        #      that may include tool_calls, tool_call_id, reasoning, etc.
        #      - These must be passed through intact so the API sees valid
        #        assistant→tool sequences (dropping tool_calls causes 500 errors)
        #
        # Telegram observed group context is handled structurally here:
        # observed=True transcript rows are withheld from replayable
        # history and attached to the current addressed message as
        # API-only context, so persisted history stores only the real
        # addressed user turn.
        agent_history, observed_group_context = _build_gateway_agent_history(
            ctx.history,
            channel_prompt=ctx.channel_prompt,
            inject_timestamps=_message_timestamps_enabled(ctx.user_config),
        )

        # FTS write-corruption guard (#50502): when message persistence
        # fails silently through corrupt FTS triggers, the reloaded
        # transcript above is stale/empty even though the SAME cached agent
        # still holds the full live conversation in `_session_messages`.
        # Replacing the live transcript with that shorter copy causes
        # immediate same-session amnesia. Only applies when we reused a
        # cached agent bound to this exact session_id.
        if reused_cached_agent and getattr(agent, "session_id", None) == ctx.session_id:
            _selected = _select_cached_agent_history(
                agent_history, getattr(agent, "_session_messages", None)
            )
            if _selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    ctx.session_key, len(agent_history), len(_selected),
                )
                # The live in-memory history bypassed the
                # _build_gateway_agent_history cleanup pipeline above —
                # re-apply the stale-confirmation expiry (#59607) so a
                # dangerous confirmation can't slip through this path
                # either. Idempotent; messages without timestamps are
                # untouched.
                agent_history = strip_stale_dangerous_confirmations(
                    _selected, now=time.time()
                )

        # Collect MEDIA paths already in history so we can exclude them
        # from the current turn's extraction. This is compression-safe:
        # even if the message list shrinks, we know which paths are old.
        _history_media_paths: set = _collect_history_media_paths(agent_history)

        # Register per-session gateway approval callback so dangerous
        # command approval blocks the agent thread (mirrors CLI input()).
        # The callback bridges sync→async to send the approval request
        # to the user immediately.
        from tools.approval import (
            register_gateway_notify,
            reset_current_session_key,
            set_current_session_key,
            unregister_gateway_notify,
        )

        def _approval_notify_sync(approval_data: dict) -> None:
            """Send the approval request to the user from the agent thread.

                If the adapter supports interactive button-based approvals
                (e.g. Discord's ``send_exec_approval``), use that for a richer
                UX.  Otherwise fall back to a plain text message with
                ``/approve`` instructions.
                """
            # Pause the typing indicator while the agent waits for
            # user approval.  Critical for Slack's Assistant API where
            # assistant_threads_setStatus disables the compose box — the
            # user literally cannot type /approve while "is thinking..."
            # is active.  The approval message send auto-clears the Slack
            # status; pausing prevents _keep_typing from re-setting it.
            # Typing resumes in _handle_approve_command/_handle_deny_command.
            ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)

            cmd = approval_data.get("command", "")
            desc = approval_data.get("description", "dangerous command")

            # Redact credentials from the command before displaying it in
            # the approval prompt — Tirith's findings are already redacted,
            # but the raw command string still leaks secrets to the chat
            # platform (#48456). Applied here so BOTH the button-based
            # (send_exec_approval) and plain-text fallback paths below use
            # the redacted value.
            cmd = _redact_approval_command(cmd)

            # Prefer button-based approval when the adapter supports it.
            # Check the *class* for the method, not the instance — avoids
            # false positives from MagicMock auto-attribute creation in tests.
            if getattr(type(ctx._status_adapter), "send_exec_approval", None) is not None:
                try:
                    _approval_fut = safe_schedule_threadsafe(
                        ctx._status_adapter.send_exec_approval(
                            chat_id=ctx._status_chat_id,
                            command=cmd,
                            session_key=_approval_session_key,
                            description=desc,
                            metadata=ctx._status_thread_metadata,
                            allow_permanent=approval_data.get("allow_permanent", True),
                            allow_session=approval_data.get("allow_session", True),
                            smart_denied=approval_data.get("smart_denied", False),
                        ),
                        ctx._loop_for_step,
                        logger=logger,
                        log_message="send_exec_approval scheduling error",
                    )
                    if _approval_fut is None:
                        raise RuntimeError("send_exec_approval: loop unavailable")
                    _outcome = _approval_send_outcome(_approval_fut, timeout=15)
                    if _outcome == "sent":
                        return
                    if _outcome == "ambiguous":
                        # Timeout ≠ failure: the card may have posted with a
                        # late ack (slow platform API call or transient
                        # connector backpressure). The prompt
                        # registration stays alive, so a tap on the rendered
                        # card still resolves; re-sending here is what
                        # produced duplicate cards and an orphaned
                        # "/approve: nothing pending" in live relay testing.
                        # Skip the text fallback.
                        logger.warning(
                            "Button-based approval send timed out — treating "
                            "as possibly-delivered (no re-send; the prompt "
                            "stays armed for a late tap)"
                        )
                        return
                    logger.warning(
                        "Button-based approval failed (send returned error), falling back to text"
                    )
                except Exception as _e:
                    logger.warning(
                        "Button-based approval failed, falling back to text: %s", _e
                    )

            # Fallback: plain text approval prompt.  Use the adapter's
            # typed prefix so Slack/Matrix users are told the form they
            # can actually type (`!approve`) — typed "/" is blocked in
            # Slack threads and reserved by Matrix clients.
            _p = getattr(ctx._status_adapter, "typed_command_prefix", "/")
            msg = _format_exec_approval_fallback(
                cmd,
                desc,
                _p,
                allow_permanent=approval_data.get("allow_permanent", True),
                allow_session=approval_data.get("allow_session", True),
                smart_denied=approval_data.get("smart_denied", False),
            )
            try:
                _approval_send_fut = safe_schedule_threadsafe(
                    ctx._status_adapter.send(
                        ctx._status_chat_id,
                        msg,
                        metadata=_interim_metadata(ctx._status_thread_metadata),
                    ),
                    ctx._loop_for_step,
                    logger=logger,
                    log_message="Approval text-send scheduling error",
                )
                if _approval_send_fut is not None:
                    _approval_send_fut.result(timeout=15)
            except Exception as _e:
                logger.error("Failed to send approval request: %s", _e)

        # Keep real user text separate from API-only recovery guidance.  If
        # an auto-continue note is prepended below, persist the original
        # message so stale guidance never replays as user-authored text.
        _persist_user_message_override: Optional[Any] = ctx.persist_user_message
        _persist_user_timestamp_override: Optional[float] = ctx.persist_user_timestamp

        # Prepend pending model switch note so the model knows about the switch
        _pending_notes = getattr(self._runner, '_pending_model_notes', {})
        _msn = _pending_notes.pop(ctx.session_key, None) if ctx.session_key else None
        if _msn:
            ctx.message = _msn + "\n\n" + ctx.message

        # Auto-continue: if the loaded history ends with a tool result,
        # the previous agent turn was interrupted mid-work (gateway
        # restart, crash, SIGTERM).  Prepend a system note so the model
        # finishes processing the pending tool results before addressing
        # the user's new message.  (#4493)
        #
        # Session-level resume_pending (set on drain-timeout shutdown)
        # escalates the wording — the transcript's last role may be
        # anything (tool, assistant with unfinished work, etc.), so we
        # give a stronger, reason-aware instruction that subsumes the
        # tool-tail case.
        #
        # Freshness gate (#16802): both branches are gated on the age
        # of the last persisted transcript row.  That is the correct
        # "when did we last do anything here" signal for both the
        # resume_pending path (restart watchdog) and the tool-tail
        # path (in-flight tool loop killed).  We read ``history[-1]``
        # here because ``agent_history`` has already stripped the
        # ``timestamp`` field off tool/tool_call rows for API purity
        # (see the `k != "timestamp"` filter above).  Rows without a
        # timestamp (legacy transcripts) are treated as fresh so the
        # historical auto-continue behaviour is preserved.
        _freshness_window = _auto_continue_freshness_window()
        _interruption_is_fresh = _is_fresh_gateway_interruption(
            _last_transcript_timestamp(ctx.history),
            window_secs=_freshness_window,
        )

        _resume_entry = None
        if ctx.session_key:
            try:
                _resume_entry = self._runner.session_store._entries.get(ctx.session_key)
            except Exception:
                _resume_entry = None

        # resume_pending freshness uses a SECOND signal in addition to the
        # transcript clock above.  The restart watchdog stamps the session
        # with ``last_resume_marked_at`` at interrupt time — that is the
        # correct "when were we interrupted" signal.  The transcript clock
        # (_interruption_is_fresh) can be far older: an active thread you
        # return to may have its last persisted row hours back, even though
        # the interruption itself just happened.  Gating resume_pending on
        # the transcript clock alone makes the recovery note silently drop,
        # and because the startup auto-resume turn carries empty text
        # (_schedule_resume_pending_sessions), the model then receives a
        # blank user message and replies with confused "the message came
        # through blank" noise.  Treat the marker as fresh when
        # EITHER signal is fresh so the two freshness checks agree.
        _resume_mark_is_fresh = False
        if _resume_entry is not None and getattr(_resume_entry, "resume_pending", False):
            _resume_mark_is_fresh = _is_fresh_gateway_interruption(
                getattr(_resume_entry, "last_resume_marked_at", None),
                window_secs=_freshness_window,
            )
        _is_resume_pending = bool(
            _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
            and (_interruption_is_fresh or _resume_mark_is_fresh)
        )
        _has_fresh_tool_tail = bool(
            agent_history
            and agent_history[-1].get("role") == "tool"
            and _interruption_is_fresh
        )

        if _is_resume_pending:
            _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            # The empty-message case is the auto-resume startup turn
            # synthesized by _schedule_resume_pending_sessions — there is
            # no NEW user message to address.  Guidance is adapter-aware:
            # interactive platforms report the restore and ask what next;
            # non-interactive event platforms (webhook, API server)
            # continue the interrupted work instead, because nobody is
            # present to answer and an acknowledgement would silently
            # abandon the task (#57056).
            _resume_adapter = self._runner._adapter_for_source(ctx.source)
            _interactive_resume = bool(
                getattr(_resume_adapter, "interactive_resume", True)
            )
            ctx.message, _persist_user_message_override = _prepare_resume_pending_message(
                _reason, ctx.message, interactive=_interactive_resume,
            )
        elif _has_fresh_tool_tail:
            _persist_user_message_override = ctx.message
            ctx.message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + ctx.message
            )

        # Consume one-shot /reload-skills note (if the user ran
        # /reload-skills since their last turn in this session). Same
        # queue pattern as CLI: prepend to the NEXT user message, then
        # clear. Nothing was written to the transcript out-of-band, so
        # message alternation stays intact.
        _pending_notes = getattr(self._runner, "_pending_skills_reload_notes", None)
        if _pending_notes and ctx.session_key and ctx.session_key in _pending_notes:
            _srn = _pending_notes.pop(ctx.session_key, None)
            if _srn:
                ctx.message = _srn + "\n\n" + ctx.message

        # Safety net: a startup auto-resume event carries empty
        # text and relies on the resume_pending branch above to supply the
        # recovery note.  If that branch did not fire for any reason (e.g.
        # both freshness signals disagreed, or the marker was cleared
        # between scheduling and dispatch) we must NOT hand the model a
        # blank user turn — it responds with confused "the message came
        # through blank" noise.  Restricted to resume_pending sessions so
        # legitimately empty user turns (e.g. an image with no caption,
        # wrapped as native content below) are untouched.
        if (
            isinstance(ctx.message, str)
            and not ctx.message.strip()
            and _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
        ):
            _sn_reason = (
                getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            )
            _sn_adapter = self._runner._adapter_for_source(ctx.source)
            ctx.message = build_resume_recovery_note(
                _sn_reason,
                "",
                interactive=bool(
                    getattr(_sn_adapter, "interactive_resume", True)
                ),
            )

        _approval_session_key = ctx.session_key or ""
        _approval_session_token = set_current_session_key(_approval_session_key)
        register_gateway_notify(_approval_session_key, _approval_notify_sync)
        try:
            # If _prepare_inbound_message_text buffered image paths for native
            # attachment, wrap the user turn as an OpenAI-style multimodal
            # content list. Consume-and-clear so subsequent turns on the same
            # runner instance don't re-attach stale images.
            _native_imgs = self._runner._consume_pending_native_image_paths(ctx.session_key)
            if _native_imgs:
                try:
                    from agent.image_routing import build_native_content_parts
                    _parts, _skipped = build_native_content_parts(
                        ctx.message,
                        _native_imgs,
                    )
                    if _skipped:
                        logger.warning(
                            "Native image attachment: skipped %d unreadable path(s): %s",
                            len(_skipped), _skipped,
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _run_message: Any = _parts
                    else:
                        # All images failed to read — fall back to plain text.
                        _run_message = ctx.message
                except Exception as _img_exc:
                    logger.warning(
                        "Native image attachment failed, falling back to text: %s",
                        _img_exc,
                    )
                    _run_message = ctx.message
            else:
                _run_message = ctx.message

            _api_run_message = _wrap_current_message_with_observed_context(
                _run_message,
                observed_group_context,
            )
            _conversation_kwargs = {
                "conversation_history": agent_history,
                "task_id": ctx.session_id,
            }
            if _persist_user_message_override is not None:
                _conversation_kwargs["persist_user_message"] = _persist_user_message_override
            elif observed_group_context:
                _conversation_kwargs["persist_user_message"] = ctx.message
            if ctx.persist_user_display_kind:
                # Internal self-injected turn (#82888): type the persisted user
                # row at turn start so UIs render it as a timeline notice, not
                # a user bubble. Role/content are untouched and the key is
                # stripped from provider-bound payloads in conversation_loop.
                _conversation_kwargs["persist_user_display_kind"] = (
                    ctx.persist_user_display_kind
                )
            if ctx.moa_config is not None:
                _conversation_kwargs["moa_config"] = ctx.moa_config
            if _persist_user_timestamp_override is not None:
                _conversation_kwargs["persist_user_timestamp"] = _persist_user_timestamp_override
            result = agent.run_conversation(_api_run_message, **_conversation_kwargs)
        finally:
            unregister_gateway_notify(_approval_session_key)
            # Cancel any pending clarify entries so blocked agent
            # threads don't hang past the end of the run (interrupt,
            # completion, gateway shutdown).  Idempotent.
            try:
                from tools.clarify_gateway import clear_session as _clear_clarify_session
                _clear_clarify_session(_approval_session_key)
            except Exception:
                pass
            reset_current_session_key(_approval_session_token)
        # Canonicalize an explicitly emitted computer-use screenshot path at
        # the common result boundary. The streaming finalizer below and the
        # normal non-streaming delivery path must see the same response;
        # repairing only during later media scanning leaves streaming with the
        # model-mangled path and a rejected attachment.
        if isinstance(result, dict):
            _result_final = result.get("final_response")
            if isinstance(_result_final, str):
                result["final_response"] = repair_explicit_computer_use_media_paths(
                    _result_final,
                    result.get("messages", []),
                    history_offset=len(agent_history),
                )

        ctx.result_holder[0] = result

        # Signal the stream consumer that the agent is done. Pass the
        # completed final_response as the authoritative finalize payload:
        # it includes post-stream augmentation (file-mutation verifier
        # footer, turn-completion explainer) the consumer's accumulator
        # never saw, so the seal/final edit delivers the TRUE final and no
        # separate corrective send fires (live finding #11). Failed turns
        # pass nothing — error text is delivered by the gateway's normal
        # path, not baked into the stream.
        if _stream_consumer is not None:
            _final_for_stream = None
            # Adopt ONLY a genuinely completed final (review B6): interrupt
            # paths return {interrupted: True, completed: False} with a
            # DIAGNOSTIC final_response ("Operation interrupted during …")
            # and no failed key — adopting that would seal the user's
            # streamed partial answer over with the diagnostic AND make
            # delivered_final_matches reconcile, suppressing the gateway's
            # own error-delivery path. Writers of these shapes:
            # agent/conversation_loop.py interrupt/retry-abort returns.
            if (
                isinstance(result, dict)
                and not result.get("failed")
                and not result.get("interrupted")
                and result.get("completed") is not False
            ):
                _fr = result.get("final_response")
                if isinstance(_fr, str) and _fr.strip() and _fr != "(empty)":
                    _final_for_stream = _fr
            if _final_for_stream is not None:
                # Duck-type safe: test doubles / older consumers may expose a
                # zero-arg finish(). The payload is an optimization, not a
                # requirement — fall back to the bare signal.
                try:
                    _stream_consumer.finish(_final_for_stream)
                except TypeError:
                    _stream_consumer.finish()
            else:
                _stream_consumer.finish()

        # Signal the streaming-TTS consumer that the agent is done (#60671).
        # finish() is called from the outer event-loop thread after the
        # executor returns, so early returns from run_sync are also
        # finalised.  See the outer finally/completion section below.

        # Return final response, or a message if something went wrong
        final_response = result.get("final_response")

        # Extract actual token counts from the agent instance used for this run
        _last_prompt_toks = 0
        _input_toks = 0
        _output_toks = 0
        _context_length = 0
        _agent = ctx.agent_holder[0]
        if _agent and hasattr(_agent, "context_compressor"):
            _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
            _input_toks = getattr(_agent, "session_prompt_tokens", 0)
            _output_toks = getattr(_agent, "session_completion_tokens", 0)
            _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
        _resolved_model = getattr(_agent, "model", None) if _agent else None

        # Sync session_id immediately after run_conversation(). Compression
        # can rotate before a follow-up model call fails; the failure return
        # below must still point the gateway at the compressed child.
        agent = ctx.agent_holder[0]
        _session_was_split = False
        # In-place compaction (compression.in_place / #38763) compacts the
        # transcript WITHOUT rotating the id, so the id-change diff below
        # can't detect it. compress_context() sets this rotation-independent
        # flag on the agent; the gateway uses it to re-baseline transcript
        # handling (history_offset=0 + rewrite the JSONL transcript) the
        # same way a split would, even though the session_id is unchanged.
        _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', ctx.session_id) if agent else ctx.session_id
        if agent and ctx.session_key and agent_session_id != ctx.session_id:
            _session_was_split = True
            logger.info(
                "Session split detected: %s → %s (compression)",
                ctx.session_id, agent_session_id,
            )
            entry = self._runner.session_store._entries.get(ctx.session_key)
            _session_split_entry_persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not ctx._run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        ctx.session_key or "?",
                        ctx.run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    _session_split_entry_persisted = True
                elif entry_session_id != ctx.session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        ctx.session_key or "?",
                        ctx.session_id,
                        entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    self._runner.session_store._save()
                    self._runner.session_store._record_gateway_session_peer(
                        agent_session_id,
                        ctx.session_key,
                        ctx.source,
                    )
                    _session_split_entry_persisted = True

            # If this is a Telegram DM and source.thread_id was lost during
            # the session split (synthetic / recovered event), restore it
            # from the binding so _thread_metadata_for_source produces the
            # correct message_thread_id instead of routing to the General
            # thread.  Failure here is non-fatal — we log and continue;
            # worst case the message lands in General, which is the
            # pre-fix behaviour. Only do this after this run successfully
            # published its session split; a stale /stop→/new predecessor
            # must not mutate routing/binding state for the fresh session.
            if _session_split_entry_persisted and (
                getattr(ctx.source, "platform", None) == Platform.TELEGRAM
                and getattr(ctx.source, "chat_type", None) == "dm"
                and getattr(ctx.source, "thread_id", None) is None
                and self._runner._session_db is not None
            ):
                try:
                    # run_sync is off-loop (executor); sync DB is fine.
                    _binding = self._runner._session_db._db.get_telegram_topic_binding_by_session(
                        session_id=agent_session_id,
                    )
                    if _binding and _binding.get("thread_id"):
                        ctx.source.thread_id = str(_binding["thread_id"])
                        logger.debug(
                            "Restored source.thread_id=%s from binding after session split %s → %s",
                            ctx.source.thread_id,
                            ctx.session_id,
                            agent_session_id,
                        )
                except Exception:
                    logger.debug(
                        "Failed to restore thread_id from binding after session split",
                        exc_info=True,
                    )
            if _session_split_entry_persisted:
                self._runner._sync_telegram_topic_binding(
                    ctx.source, entry, reason="agent-run-compression",
                )

        effective_session_id = agent_session_id
        self._runner._sync_session_model_from_agent(effective_session_id, agent)
        # history_offset=0 whenever the agent's message list no longer has
        # the original history prefix — i.e. on rotation (split) OR in-place
        # compaction. In both cases the returned `messages` is the compacted
        # set, so the gateway must persist all of it (offset 0), not slice
        # past the pre-compaction length (which would drop everything).
        _effective_history_offset = (
            0 if (_session_was_split or _compacted_in_place) else len(agent_history)
        )

        if not final_response:
            final_response = _normalize_empty_agent_response(
                result, final_response or "", history_len=len(agent_history),
            )
            final_response = _sanitize_gateway_final_response(ctx.source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            return {
                "final_response": final_response,
                "messages": result.get("messages", []),
                "api_calls": result.get("api_calls", 0),
                "failed": result.get("failed", False),
                # Sibling of the non-empty-response return below (#64686):
                # the classifier's failure_reason must survive the
                # empty-response normalization path too, or downstream
                # consumers (TUI billing surface, transient-failure
                # persistence) lose the structured reason exactly when
                # the run produced no text.
                "failure_reason": result.get("failure_reason"),
                "partial": result.get("partial", False),
                "completed": result.get("completed"),
                "interrupted": result.get("interrupted", False),
                "interrupt_message": result.get("interrupt_message"),
                "error": result.get("error"),
                "compression_exhausted": result.get("compression_exhausted", False),
                "compression_deferred": result.get("compression_deferred", False),
                "tools": ctx.tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "compacted_in_place": _compacted_in_place,
                "session_id": effective_session_id,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
            }

        # Scan tool results for MEDIA:<path> tags that need to be delivered
        # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
        # in its JSON response, but the model's final text reply usually
        # doesn't include them.  We collect unique tags from tool results and
        # append any that aren't already present in the final response, so the
        # adapter's extract_media() can find and deliver the files exactly once.
        #
        # Scope the scan to THIS turn's tool results only. ``agent_history``
        # was passed into run_conversation as ``conversation_history``, so the
        # agent's returned ``messages`` list is ``agent_history`` followed by
        # the messages produced this turn. Slicing at ``len(agent_history)``
        # isolates the current turn precisely, so a stale MEDIA: path emitted
        # by a tool several turns earlier (still present in the full message
        # list) can never leak onto a later text-only reply. (Fixes #34608)
        #
        # Path-based deduplication against _history_media_paths (collected
        # before run_conversation) is retained as a secondary guard. It is
        # also the sole guard on the fallback branch taken when mid-run
        # context compression shrinks the message list below the original
        # history length, preserving the compression-safe behaviour of #160.
        if "MEDIA:" not in final_response:
            media_tags, has_voice_directive = _collect_auto_append_media_tags(
                result.get("messages", []),
                history_offset=len(agent_history),
                history_media_paths=_history_media_paths,
            )

            if media_tags:
                seen = set()
                unique_tags = []
                for tag in media_tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                if has_voice_directive:
                    unique_tags.insert(0, "[[audio_as_voice]]")
                final_response = final_response + "\n" + "\n".join(unique_tags)

        # Auto-titling runs at TURN START (agent/turn_context.py) from the
        # user's message alone, so it no longer waits on final_response — a
        # failed or interrupted turn still gets a titled session. The
        # platform-specific thread-rename callbacks are attached to the agent
        # as `_on_session_title` before the run starts (see
        # _attach_session_title_callback), because the titler now fires from
        # inside the turn prologue rather than from here.

        return {
            "final_response": final_response,
            "last_reasoning": result.get("last_reasoning"),
            "messages": ctx.result_holder[0].get("messages", []) if ctx.result_holder[0] else [],
            "api_calls": ctx.result_holder[0].get("api_calls", 0) if ctx.result_holder[0] else 0,
            "failed": ctx.result_holder[0].get("failed", False) if ctx.result_holder[0] else False,
            "failure_reason": (
                ctx.result_holder[0].get("failure_reason") if ctx.result_holder[0] else None
            ),
            "completed": ctx.result_holder[0].get("completed") if ctx.result_holder[0] else None,
            "interrupted": ctx.result_holder[0].get("interrupted", False) if ctx.result_holder[0] else False,
            "partial": ctx.result_holder[0].get("partial", False) if ctx.result_holder[0] else False,
            "error": ctx.result_holder[0].get("error") if ctx.result_holder[0] else None,
            "interrupt_message": ctx.result_holder[0].get("interrupt_message") if ctx.result_holder[0] else None,
            "compression_exhausted": (
                ctx.result_holder[0].get("compression_exhausted", False)
                if ctx.result_holder[0] else False
            ),
            # Soft lock-contention defer (#69870 consumer): distinct from
            # compression_exhausted so the gateway never auto-resets a
            # session that a concurrent compressor is about to shrink.
            "compression_deferred": (
                ctx.result_holder[0].get("compression_deferred", False)
                if ctx.result_holder[0] else False
            ),
            "tools": ctx.tools_holder[0] or [],
            "history_offset": _effective_history_offset,
            "compacted_in_place": _compacted_in_place,
            "last_prompt_tokens": _last_prompt_toks,
            "input_tokens": _input_toks,
            "output_tokens": _output_toks,
            "model": _resolved_model,
            "context_length": _context_length,
            "session_id": effective_session_id,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Pass through the agent_persisted flag so the persistence block
            # above can correctly determine whether the codex app-server path
            # self-persisted (it didn't — see codex_runtime.py).  Default
            # True preserves the skip-db behaviour for the standard runtime.
            "agent_persisted": (ctx.result_holder[0].get("agent_persisted", True) if ctx.result_holder[0] else True),
        }



# Sentinel for "no explicit session DB has been pinned on this runner", so the
# ``_session_db`` property can distinguish "resolve from the active profile
# scope" from a deliberate ``runner._session_db = None`` (which disables
# DB-backed commands and is how many suites construct a bare runner).  A plain
# ``None`` cannot express both.  Mirrors ``gateway.session._DB_UNPINNED``.
# "No session DB pinned": lets ``_session_db`` distinguish "resolve from profile scope" from a
# deliberate ``runner._session_db = None`` (disables DB commands). Mirrors gateway.session._DB_UNPINNED.
_SESSION_DB_UNPINNED = object()


# Only explicit suspension can replace a routed conversation.
_AUTO_RESET_CONTEXT_NOTES = {
    "suspended": "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]",
}


def _write_runtime_status_quiet(**fields: Any) -> None:
    """Best-effort ``gateway_state.json`` write; status persistence must never abort the caller."""
    try:
        from gateway.status import write_runtime_status
        write_runtime_status(**fields)
    except Exception:
        pass


def _command_origin_for_source(source: Any) -> Optional[dict]:
    """Delivery origin for a shared CLI/gateway command so its job replies to this chat/thread."""
    try:
        platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
        chat_id = getattr(source, "chat_id", None)
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": str(chat_id),
                "chat_name": getattr(source, "chat_name", None),
                "thread_id": getattr(source, "thread_id", None)}
    except Exception:
        pass
    return None


def _builtin_adapter_import(module: str, adapter_name: str, requirement: str):
    """Lazy-import ``(adapter_cls, requirements_ok)`` from ``gateway.platforms.<module>``."""
    import importlib
    mod = importlib.import_module(f"gateway.platforms.{module}")
    return getattr(mod, adapter_name), getattr(mod, requirement)


# platform -> (module, adapter class, requirements probe, warning on probe failure).
_BUILTIN_ADAPTERS: dict[Platform, tuple[str, str, str, str]] = {
    Platform.WHATSAPP_CLOUD: ("whatsapp_cloud", "WhatsAppCloudAdapter", "check_whatsapp_cloud_requirements",
                              "WhatsApp Cloud: aiohttp/httpx missing — reinstall hermes-agent"),
    Platform.SIGNAL: ("signal", "SignalAdapter", "check_signal_requirements",
                      "Signal: runtime requirements not met"),
    Platform.WEIXIN: ("weixin", "WeixinAdapter", "check_weixin_requirements",
                      "Weixin: aiohttp/cryptography not installed"),
    Platform.API_SERVER: ("api_server", "APIServerAdapter", "check_api_server_requirements",
                          "API Server: aiohttp not installed"),
    Platform.WEBHOOK: ("webhook", "WebhookAdapter", "check_webhook_requirements",
                       "Webhook: aiohttp not installed"),
    Platform.MSGRAPH_WEBHOOK: ("msgraph_webhook", "MSGraphWebhookAdapter", "check_msgraph_webhook_requirements",
                               "MSGraph webhook: aiohttp not installed"),
    Platform.BLUEBUBBLES: ("bluebubbles", "BlueBubblesAdapter", "check_bluebubbles_requirements",
                           "BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured"),
    Platform.QQBOT: ("qqbot", "QQAdapter", "check_qq_requirements",
                     "QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured"),
    Platform.YUANBAO: ("yuanbao", "YuanbaoAdapter", "WEBSOCKETS_AVAILABLE",
                       "Yuanbao: websockets not installed. Run: pip install websockets")}


def _instantiate_builtin_adapter(platform: Platform, config: Any) -> Optional[BasePlatformAdapter]:
    """Instantiate a core (non-plugin) adapter, or None when its requirements are unmet/unknown."""
    spec = _BUILTIN_ADAPTERS.get(platform)
    if spec is None:
        return None
    module, adapter_name, requirement, warning = spec
    adapter_cls, requirements_ok = _builtin_adapter_import(module, adapter_name, requirement)
    if not (requirements_ok() if callable(requirements_ok) else requirements_ok):
        logger.warning(warning)
        return None
    if platform == Platform.SIGNAL:
        from gateway.platforms.signal import validate_signal_config
        if not validate_signal_config(config):
            logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
            return None
    return adapter_cls(config)


class GatewayRunner(
    GatewayAuthorizationMixin, GatewayKanbanWatchersMixin, GatewaySlashCommandsMixin,
    GatewayVoiceMixin, GatewayAdapterLifecycleMixin, GatewayTopicThreadsMixin, GatewayTurnMixin,
    GatewayShutdownMixin, GatewayBusySessionMixin, GatewayConfigLoadersMixin, GatewayStartupMixin,
    GatewaySessionWatchersMixin, GatewayNotificationsMixin, GatewayInboundMixin, GatewayGoalsMixin,
    GatewayAgentCacheMixin):
    """Main gateway controller: manages adapter lifecycles, routes messages to/from the agent."""

    # Class-level defaults so partial construction in tests doesn't blow up on attribute access.
    _busy_input_mode: str = "interrupt"
    _busy_text_mode: str = "interrupt"
    _restart_drain_timeout: float = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    _restart_after_turn_timeout: float = DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    _cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
    _signal_interrupt_grace_timeout: float = DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    _exit_code: Optional[int] = None
    _draining: bool = False
    _external_drain_active: bool = False
    _restart_requested: bool = False
    _restart_task_started: bool = False
    _restart_detached: bool = False
    _restart_via_service: bool = False
    _detached_restart_helper_started: bool = False
    _restart_command_source: Optional[SessionSource] = None
    _stop_task: Optional[asyncio.Task] = None
    _restart_task: Optional[asyncio.Task] = None
    _profile_failed_platforms: Optional[Dict[str, Dict[Platform, asyncio.Task]]] = None
    _systemd_watchdog: Optional[Any] = None
    _startup_restore_in_progress: bool = False
    _startup_warmup_task: Optional[asyncio.Task] = None

    # Legacy per-session dict attrs as LIVE views over ``self._sessions``; new code: _session_state(key)
    _running_agents = legacy_dict_property("_running_agents")
    _running_agents_ts = legacy_dict_property("_running_agents_ts")
    _active_session_leases = legacy_dict_property("_active_session_leases")
    _busy_ack_ts = legacy_dict_property("_busy_ack_ts")
    _turn_lease_tokens = legacy_lease_token_property()
    _session_run_generation = legacy_dict_property("_session_run_generation")
    _session_model_overrides = legacy_dict_property("_session_model_overrides")
    _pending_one_turn_model_restores = legacy_dict_property("_pending_one_turn_model_restores")
    _session_reasoning_overrides = legacy_dict_property("_session_reasoning_overrides")
    _session_service_tier_overrides = legacy_dict_property("_session_service_tier_overrides")
    _last_resolved_model = legacy_dict_property("_last_resolved_model")
    _queued_events = legacy_dict_property("_queued_events")
    _pending_turn_sidecar_notes = legacy_dict_property("_pending_turn_sidecar_notes")
    _pending_messages = legacy_dict_property("_pending_messages")
    _pending_native_image_paths_by_session = legacy_dict_property(
        "_pending_native_image_paths_by_session")
    _session_ephemeral_pin = legacy_dict_property("_session_ephemeral_pin")
    _session_vc_last = legacy_dict_property("_session_vc_last")
    _pending_approvals = legacy_dict_property("_pending_approvals")
    _update_prompt_pending = legacy_dict_property("_update_prompt_pending")

    def _sessions_map(self) -> Dict[str, "SessionState"]:
        """Per-session state map; lazily created so bare ``object.__new__`` test runners work."""
        sessions = self.__dict__.get("_sessions")
        if sessions is None:
            sessions = {}
            self.__dict__["_sessions"] = sessions
        return sessions

    def _session_state(self, session_key: str) -> "SessionState":
        """Get-or-create the :class:`SessionState` for ``session_key``."""
        sessions = self._sessions_map()
        state = sessions.get(session_key)
        if state is None:
            state = SessionState()
            sessions[session_key] = state
        return state

    def _peek_session_state(self, session_key: str) -> Optional["SessionState"]:
        """Return the SessionState for ``session_key`` without creating one."""
        sessions = self.__dict__.get("_sessions")
        return sessions.get(session_key) if sessions else None

    def _is_session_running(self, session_key: str) -> bool:
        """True when the session holds a running-turn slot (agent or sentinel)."""
        state = self._peek_session_state(session_key)
        return state is not None and state.turn.agent is not None

    def _running_agent_items(self) -> List[tuple]:
        """(session_key, agent) pairs for sessions with a running turn (incl. pending sentinels)."""
        return [(key, state.turn.agent) for key, state in self._sessions_map().items()
                if state.turn.agent is not None]
    # Loop-liveness / watchdog handles; class-level defaults so partially constructed test runners work.
    # Class-level defaults so partial construction in tests doesn't blow up on access; the real values are
    # set in __init__ / start() / stop(). See #66892, #69089.
    _loop_heartbeat_task: Optional["asyncio.Task"] = None
    _loop_floor_timer_handle: Optional[Any] = None
    _loop_liveness_watchdog: Optional[Any] = None
    _gateway_started_at: float = 0.0
    _shutdown_watchdog_done: Optional["threading.Event"] = None
    _platform_lock_takeover_on_start: bool = False
    _reconnect_watcher_task: Optional["asyncio.Task"] = None

    def __init__(self, config: Optional[GatewayConfig] = None):
        global _gateway_runner_ref
        # With multiplex_profiles on, load under the default profile secret scope so bot tokens in its
        # .env resolve as secondary profiles' do; explicit config= injection (tests) is left untouched.
        # See #64674.
        self.config = config if config is not None else load_gateway_config_for_runner()
        # Multiplexer flag flips agent.secret_scope.get_secret() to fail-closed on unscoped credential
        # reads, so a missed migration crashes loudly instead of leaking a cross-profile value.
        try:
            from agent.secret_scope import set_multiplex_active
            set_multiplex_active(bool(getattr(self.config, "multiplex_profiles", False)))
        except Exception:
            logger.debug("could not set multiplex-active flag", exc_info=True)
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        # Non-None means SessionDB init failed — the gateway broadcasts a one-time warning to the home
        # channel(s) after connecting so the user learns persistence is broken before /resume fails.
        # See #88235.
        self._session_db_init_error: Optional[str] = None
        # Non-default profiles' adapters by profile then Platform; self.adapters stays the default's map.
        self._profile_adapters: Dict[str, Dict[Platform, BasePlatformAdapter]] = {}
        self._warn_if_docker_media_delivery_is_risky()
        _gateway_runner_ref = _weakref.ref(self)

        self._init_runtime_settings()
        self._init_session_store()
        self._init_lifecycle_state()
        self._init_runtime_caches()
        self._init_startup_checks()
        self._init_session_db()
        self._init_registries_and_clocks()

    def _init_runtime_settings(self) -> None:
        """Load ephemeral per-call config (prefill, reasoning, busy modes, timeouts, routing)."""
        self._prefill_messages = self._load_prefill_messages()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._busy_text_mode = self._load_busy_text_mode()
        # Secondary-profile busy modes snapshotted at multiplex startup; handlers never reread config.
        self._busy_input_modes_by_profile: Dict[str, str] = {}
        self._busy_text_modes_by_profile: Dict[str, str] = {}
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._restart_after_turn_timeout = self._load_restart_after_turn_timeout()
        self._cron_drain_timeout = self._load_cron_drain_timeout()
        self._signal_interrupt_grace_timeout = self._load_signal_interrupt_grace_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()

    def _init_session_store(self) -> None:
        """Build the SessionStore (with process-registry reset guard), its async facade and the router."""
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(
                key))
        # Loop-side boundary: sync helpers use ``session_store`` directly; async handlers await this facade.
        self._async_session_store = AsyncSessionStore(self.session_store)
        self.delivery_router = DeliveryRouter(self.config)

    def _init_lifecycle_state(self) -> None:
        """Initialise run/exit/restart flags, per-session state, and completion-delivery bookkeeping."""
        self._running = self._exit_cleanly = self._exit_with_failure = self._draining = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._profile_failed_platforms: Dict[str, Dict[Platform, asyncio.Task]] = {}
        self._systemd_watchdog = None
        # External (NAS-driven) drain, distinct from one-way ``_draining``: set while ``.drain_request.json``
        # exists — NEW turns refused, process stays up, removing the marker reverts to ``running``.
        self._external_drain_active = False
        # ``_signal_initiated_shutdown``: SIGTERM/SIGINT with no planned-stop/takeover marker (container,
        # OOM, bare kill); _stop_impl must NOT persist gateway_state=stopped or container_boot won't restart.
        self._restart_requested = self._signal_initiated_shutdown = self._restart_task_started = False
        self._restart_detached = self._restart_via_service = self._detached_restart_helper_started = False
        self._restart_command_source: Optional[SessionSource] = None
        # Construction clock: bounds the /restart redelivery guard's window (missing dedup marker = stale).
        self._startup_time: float = time.time()
        # True when booted from a chat /restart (.restart_notify.json existed). One-shot signal so the
        # marker-missing fallback suppresses a /restart only when we KNOW we just restarted.
        self._booted_from_restart: bool = False
        self._stop_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._executor_lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set on gateway stop so the recreate-on-shutdown path can't resurrect the pool.
        self._executor_closing = False
        # ALL per-session state lives here (gateway/session_state.py); use _session_state / _peek_session_state.
        self._sessions: Dict[str, SessionState] = {}
        # Per-SESSION_ID turn lease: serializes [load history → run → flush] when two ROUTING KEYS resolve
        # to one session_id (switch_session's many-to-one mapping), which routing-key guards cannot see.
        self._turn_leases = SessionTurnLeaseRegistry()
        # Stall-notified keys clear when pending clears / activity resumes / conversation boundary.
        # Tokens for held turn leases, keyed by (routing key, run generation) so release is granted per-turn
        # and a stale unwind can never free a newer turn's lease (#28686 ownership lesson). Held turn-lease
        # tokens live on SessionState.turn.lease_token / .lease_generation (the old dict was keyed (routing
        # key, generation) so a stale unwind could never free a newer turn's lease — the generation field
        # preserves that ownership check, #28686). Runner-level queued interrupt text lives on
        # SessionState.persistent.pending_command_text (NOTE: distinct from the adapter-level
        # _pending_messages Dict[str, MessageEvent] in gateway/platforms/base.py, which shares the legacy
        # name). Last successfully-resolved (non-empty) model, keyed by session. Used as a fallback when a
        # fresh config read transiently returns an empty model (e.g. an mtime-keyed config-cache miss during
        # a post-interrupt recovery turn). Without this, the agent is built with model="" and every API call
        # fails HTTP 400 "No models provided" — the session goes silent until the user manually re-sends.
        # See #35314. The ``"*"`` session entry holds a process-wide last-known-good for sessions seen for
        # the first time. Lives on SessionState.conversation.last_resolved_model. Overflow buffer for
        # explicit /queue commands. The adapter-level _pending_messages dict is a single slot per session
        # (designed for "next-turn" follow-ups where repeated sends collapse into one event).  /queue has
        # different semantics: each invocation must produce its own full agent turn, in FIFO order, with no
        # merging. When the slot is occupied, additional /queue items land here and are promoted
        # one-at-a-time after each run's drain. Cleared on /new and /reset.  /model and other mid-session
        # operations preserve the queue. Lives on SessionState.conversation.queued_events; native image
        # paths, busy-ack debounce timestamps and the monotonic run-generation counter (#28686, NEVER reset)
        # live on SessionState too. See gateway.session_stall.
        self._session_stall_notified: Dict[str, bool] = {}
        # Startup restore gate: while restart-interrupted sessions auto-resume, real inbound messages
        # queue instead of competing with the synthetic resume turns; drained after all resume tasks end.
        self._startup_restore_in_progress = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # Set by start_gateway() only for an explicit ``--replace`` launch; scoped to each adapter's
        # cold-start connect and removed before any reconnect can run.
        self._platform_lock_takeover_on_start = False
        # Capped LRU of live SessionSources for fallback routing (shutdown notices, synthetic events) when
        # the persisted origin is missing and _parse_session_key can't recover thread_id.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512
        # Lifecycle-scoped completion dedup: closes queue/watcher races inside one gateway without claiming
        # exactly-once across a crash; durable replay state stays owned by tools.async_delegation.
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight: set[tuple[str, str, object]] = set()
        self._completion_deliveries_delivered: "OrderedDict[tuple[str, str, object], None]" = OrderedDict()
        self._completion_delivery_retention = 2048
        # Agent-triggered terminal completions from one conversation often land in the same scheduler
        # tick; hold them briefly so the agent gets one synthetic turn instead of one per process.
        # See #70300.
        self._completion_notification_batches: dict[tuple[str, ...], list[tuple[str, dict, asyncio.Future]]] = {}
        self._completion_notification_batch_tasks: dict[tuple[str, ...], asyncio.Task] = {}
        self._completion_notification_batch_flush_tasks: set[asyncio.Task] = set()
        self._completion_notification_batch_window = 0.1
        self._completion_notification_batches_stopping = False

    def _init_runtime_caches(self) -> None:
        """Agent cache, profile identity, Teams runtime, failed-platform tracking, slash-confirm counter."""
        # AIAgent per session preserves prompt caching (fresh agent per message ~10x cost on Anthropic).
        # Value: (AIAgent, config_signature); LRU cap in _enforce_agent_cache_cap, TTL in expiry watcher.
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = threading.Lock()
        # Launch-time identity of the profile that owns ``self.adapters``; ``_authorization_adapter``
        # compares against this rather than the per-turn ``_active_profile_name()``.
        self._primary_profile_name = self._kanban_notifier_profile = self._active_profile_name()
        # Teams meeting pipeline runtime (bound later when msgraph_webhook adapter exists).
        self._teams_pipeline_runtime = None
        self._teams_pipeline_runtime_error: Optional[str] = None
        # Failed-to-connect platforms for background reconnection: Platform -> {config, attempts, next_retry}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}
        # Strong refs to detached fatal-error handler tasks so the loop can't GC them mid-run.
        self._fatal_handler_tasks: set = set()
        # Slash-confirm state lives in tools.slash_confirm (module-level) so adapters resolve callbacks
        # without a runner backref; local counter keeps confirm_ids compact (64-byte callback_data caps).
        import itertools
        self._slash_confirm_counter = itertools.count(1)

    def _init_startup_checks(self) -> None:
        """Ensure tirith is installed and warn when manual approvals have no automated assessor."""
        def _ensure_tirith() -> None:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)  # downloads if needed; fail-open at scan time

        _best_effort(_ensure_tirith)

        # Manual approvals with no automated assessor (tirith off AND no auxiliary.approval) fail closed
        # on unattended gateways — surface it so operators knowingly enable one.
        try:
            from hermes_cli.config import load_config as _load_full_config
            # Startup heads-up (#30882): a gateway in manual approval mode with no automated risk assessor
            # (tirith disabled AND no auxiliary.approval model) can only gate dangerous commands /
            # execute_code scripts via live in-chat approval.
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation.")
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)

    def _init_session_db(self) -> None:
        """Open the session DB for the active scope and run opportunistic state.db / checkpoint maintenance."""
        # Session DB is a property caching one AsyncSessionDB per path (a handle bound here would pin the
        # root home under multiplex); priming here keeps startup diagnostics at init.
        # Initialize session database for session_search tool support. Same frozen-handle class of bug as
        # SessionStore._db (#88532): a handle bound here is pinned to the process's root home, but /resume,
        # /title, /history and session search all run inside _profile_runtime_scope on a multiplexed gateway
        # and must see that profile's own state.db.
        self._session_db_pinned: Any = _SESSION_DB_UNPINNED
        self._session_db_handles: Dict[Path, Any] = {}
        self._session_db_handles_lock = threading.Lock()
        from gateway.session_db_recovery import RecoverableHandleCache
        self._session_db_handle_cache = RecoverableHandleCache(
            handles=self._session_db_handles, lock=self._session_db_handles_lock)
        try:
            self._open_session_db_for_active_scope(raise_on_error=True)
        except Exception as e:
            # WARNING (not DEBUG) so it lands in errors.log; else an NFS HERMES_HOME silently loses /resume etc.
            logger.warning("SQLite session store not available: %s", e)
            self._session_db_init_error = str(e)  # surfaced on the home channel(s) once connected

        # Opportunistic state.db maintenance (prune + optional VACUUM), at most once per min_interval_hours.
        # A few blocking seconds per day is fine for a long-lived gateway; failures log, never raise.
        # Surface the failure to the user via their home channel(s) once the gateway connects. Without this,
        # state.db corruption or NFS/SMB lock failures silently degrade the entire gateway — messages may
        # flow but nothing is persisted, and the user has no indication until they try /resume and find
        # nothing (#88235).
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_archive", False):
                    self._session_db._db.maybe_auto_archive(
                        idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
                if _sess_cfg.get("auto_prune", False):
                    # Construction-time, before the loop serves traffic; sync DB is fine.
                    self._session_db._db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        min_vacuum_interval_days=int(
                            _sess_cfg.get("min_vacuum_interval_days", 30)),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir)
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Stale checkpoint repo cleanup; opt-in via checkpoints.auto_prune, idempotent via .last_prune.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                # delete_orphans never honoured unattended: a missing workdir is ambiguous (deleted vs.
                # unmounted share); orphan cleanup is only via explicit `hermes checkpoints prune`.
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=False,
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)))
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)

    def _init_registries_and_clocks(self) -> None:
        """Pairing stores, hook registry, voice modes, background-task set, liveness and idle clocks."""
        # ``pairing_store``: global/default store (CLI, callers without profile context); ``pairing_stores``:
        # per-profile map ``authz_mixin._is_user_authorized`` routes through (one whitelist per profile).
        from gateway.pairing import PairingStore
        from gateway.hooks import HookRegistry
        self.pairing_store = PairingStore()
        self.pairing_stores: Dict[str, "PairingStore"] = {}
        self.hooks = HookRegistry()
        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        # Per-(guild,user) transcript dedup: the voice/STT pipeline can emit one utterance twice.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}
        # Background tasks kept referenced so they are not garbage-collected mid-execution.
        self._background_tasks: set = set()
        # Event-loop liveness heartbeat: rewritten every 30s while the loop dispatches; supervisors use
        # the file mtime / updated_at to tell "process alive" from "loop frozen".
        # See #66892.
        self._gateway_started_at: float = time.time()
        self._loop_heartbeat_task: Optional[asyncio.Task] = None
        self._loop_floor_timer_handle = self._loop_liveness_watchdog = None
        # scale-to-zero: gateway-scoped "last inbound seen" clock, stamped in _handle_message (the single
        # inbound chokepoint) and seeded to "now" so a fresh gateway isn't idle from epoch.
        self._last_inbound_at: float = time.time()
        # Re-arm cooldown after a wake so we don't go dormant again before the drained backlog updates
        # the clock; and a one-shot latch so the "platform owns the suspend" notice logs once.
        self._scale_to_zero_cooldown_until: float = 0.0
        self._scale_to_zero_no_suspend_logged: bool = False

    def _open_session_db_for_active_scope(self, raise_on_error: bool = False) -> Any:
        """AsyncSessionDB for the active profile scope, resolved per access (not in ``__init__``) since
        ``SessionDB()`` reads the context-local HERMES_HOME; one handle cached per path. Construction
        failure enters bounded backoff; ``raise_on_error=True`` (priming) propagates it.

        Same per-path cache as ``SessionStore._open_session_db_for_active_scope`` (#88532): ``SessionDB()``
        resolves ``_default_db_path()`` at call time through the context-local HERMES_HOME override
        installed by ``_profile_runtime_scope``, so resolving per access — instead of once in ``__init__`` —
        is what lets /resume, /title, /history and session search on a multiplexed gateway read the *serving
        profile's* store rather than the root one.
        One ``AsyncSessionDB`` is cached per resolved path, so the wrapper identity is stable per profile
        (callers compare and stash it) and two profiles never share a handle. A construction failure enters
        bounded backoff; one caller retries after the deadline while concurrent callers continue to see the
        unavailable fallback. ``raise_on_error=True`` (construction-time priming) propagates the failure
        after recording that recoverable state so ``__init__`` can record ``_session_db_init_error`` for the
        #88235 broadcast.
        """
        from hermes_state import AsyncSessionDB, _default_db_path
        from hermes_state_registry import acquire
        from gateway.session_db_recovery import RecoverableHandleCache
        path = Path(_default_db_path())
        cache = getattr(self, "_session_db_handle_cache", None)
        if cache is None:
            # Test runners built with object.__new__ skip __init__.
            cache = RecoverableHandleCache(
                handles=self._session_db_handles, lock=self._session_db_handles_lock)
            self._session_db_handle_cache = cache

        def _open():
            # Borrow the SessionStore's handle (same path) so state.db doesn't get two writers/pools.
            # The store owns/sweeps it at shutdown; this cache holds only the async wrapper (close_all).
            # Both caches resolve the SAME ``_default_db_path()``, so the process was holding two writer
            # connections and two read pools against one state.db — the fd budget doubled for nothing, and
            # doubled again per profile on a multiplexed gateway (#98573). A borrowed wrapper cannot go
            # stale in practice: the store's cache only drops handles in close_all_db_handles() (shutdown),
            # and while the store's own open is failing there is nothing to borrow, so nothing is cached
            # here either.
            store = getattr(self, "session_store", None)
            borrowed = getattr(store, "_db", None) if store is not None else None
            if borrowed is not None:
                wrapper = AsyncSessionDB(borrowed)
                # close_all_session_db_handles() must not close what the store owns (its sweep runs first).
                wrapper.__dict__["_hermes_borrowed_handle"] = True
                return wrapper
            if store is not None:
                # Store handle unavailable: opening our own would resurrect the duplicate borrowed away.
                raise RuntimeError("SessionStore SQLite handle unavailable")
            try:
                return AsyncSessionDB(acquire())
            except Exception as exc:
                logger.warning("SQLite session store not available: %s", exc)
                raise

        def _recovered() -> None:
            self._session_db_init_error = None
            logger.info("SQLite session store recovered")

        return cache.get(path, _open, raise_on_error=raise_on_error, on_recovered=_recovered)

    @property
    def _session_db(self) -> Any:
        """The AsyncSessionDB for the active profile scope, or a pinned override (assigning
        ``runner._session_db`` pins it for every later read — tests install fakes/None this way)."""
        if self._session_db_pinned is not _SESSION_DB_UNPINNED:
            return self._session_db_pinned
        return self._open_session_db_for_active_scope()

    @_session_db.setter
    def _session_db(self, value) -> None:
        self._session_db_pinned = value

    def close_all_session_db_handles(self) -> None:
        """Close every per-profile AsyncSessionDB this runner opened.

        Drained under the lock, closed outside it; a pinned handle is the pinner's to close. Wrappers
        BORROWED from ``session_store`` are skipped: the store's sweep (runs first) closes them.

        See #98573.
        """
        def _close(db) -> None:
            if getattr(db, "__dict__", {}).get("_hermes_borrowed_handle"):
                return
            inner = getattr(db, "_db", db)
            if inner is None or not hasattr(inner, "close"):
                return
            # Shared instances no-op on close() (the registry owns the lifecycle). Release the refcount
            # instead (#90837).
            from hermes_state_registry import release_or_close
            try:
                release_or_close(inner)
            except Exception as exc:
                logger.debug("SessionDB close error during handle sweep: %s", exc)

        self._session_db_handle_cache.close_all(_close)

    def _wire_teams_pipeline_runtime(self) -> None:
        """Bind the Teams meeting pipeline runtime to Graph webhook ingress (no-op if adapter/plugin off)."""
        if Platform.MSGRAPH_WEBHOOK not in self.adapters:
            return
        if not _teams_pipeline_plugin_enabled():
            logger.debug("Teams pipeline plugin is disabled; skipping runtime wiring")
            return
        try:
            from plugins.teams_pipeline.runtime import bind_gateway_runtime
        except Exception as exc:
            logger.warning("Teams pipeline runtime import failed: %s", exc)
            return
        try:
            bound = bind_gateway_runtime(self)
        except Exception as exc:
            logger.warning("Teams pipeline runtime wiring failed: %s", exc)
            return
        if bound:
            logger.info("Teams pipeline runtime bound to msgraph webhook ingress")
        elif self._teams_pipeline_runtime_error:
            logger.warning(
                "Teams pipeline runtime unavailable: %s", self._teams_pipeline_runtime_error)

    def _warn_if_docker_media_delivery_is_risky(self) -> None:
        """Warn when Docker-backed gateways lack an explicit export mount: MEDIA delivery runs in the
        gateway process, so model-emitted paths like `/output/report.txt` must be host-readable."""
        if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
            return
        connected = self.config.get_connected_platforms()
        messaging_platforms = [p for p in connected if p not in {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}]
        if not messaging_platforms:
            return

        raw_volumes = os.getenv("TERMINAL_DOCKER_VOLUMES", "").strip()
        volumes: List[str] = []
        if raw_volumes:
            try:
                parsed = json.loads(raw_volumes)
                if isinstance(parsed, list):
                    volumes = [str(v) for v in parsed if isinstance(v, str)]
            except Exception:
                logger.debug("Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning", exc_info=True)

        for spec in volumes:
            match = _DOCKER_VOLUME_SPEC_RE.match(spec)
            if match and match.group("container") in _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS:
                return
        logger.warning(
            "Docker backend is enabled for the messaging gateway but no explicit host-visible "
            "output mount (for example '/home/user/.hermes/cache/documents:/output') is configured. "
            "This is fine if the model already emits host-visible paths, but MEDIA file delivery can fail "
            "for container-local paths like '/workspace/...' or '/output/...'.")

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    should_exit_cleanly = property(lambda self: self._exit_cleanly)
    should_exit_with_failure = property(lambda self: self._exit_with_failure)
    exit_reason = property(lambda self: self._exit_reason)
    exit_code = property(lambda self: self._exit_code)

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        # Mirror SessionStore._resolve_profile_for_key so this fallback yields the primary path's
        # namespace: None (legacy agent:main) unless multiplexing is on, then the active profile.
        _profile = None
        if getattr(config, "multiplex_profiles", False):
            if source.profile:
                _profile = source.profile
            else:
                try:
                    from hermes_cli.profiles import get_active_profile_name
                    _profile = get_active_profile_name() or "default"
                except Exception:
                    _profile = None
        return build_session_key(
            source, group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
            profile=_profile)

    # Telegram General topic in forum-enabled private chats: clients omit message_thread_id or send "1"; both = root.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})
    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0

    def _normalize_source_for_session_key(self, source: SessionSource) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes. Always derive override
        storage keys from the result: ``_handle_message_with_agent`` rewrites ``thread_id`` before
        deriving the session key, so keys from the raw ``event.source`` are never read next turn.

        ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
        ``_recover_telegram_topic_thread_id`` *before* deriving the session key for a normal message turn (a
        lobby/stripped reply gets pinned to the user's last-active topic). Session-scoped command handlers
        like ``/model`` and ``/reasoning`` derive their override key from the raw inbound ``event.source``,
        which skips that recovery — so the override is stored under a different key than the next message
        turn reads, and the override is silently dropped on Telegram forum topics and after compression
        session splits (#30479).
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        return source if recovered is None else dataclasses.replace(source, thread_id=recovered)

    def _resolve_session_key_or_none(self, source, session_key: Optional[str]) -> Optional[str]:
        """``session_key`` if given, else the key for ``source`` (None when it cannot be derived)."""
        if session_key or source is None:
            return session_key
        try:
            return self._session_key_for_source(source)
        except Exception:
            return None

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        _write_runtime_status_quiet(
            gateway_state=gateway_state, exit_reason=exit_reason,
            restart_requested=self._restart_requested, active_agents=self._active_work_count())

    def _persist_active_agents(self) -> None:
        """Persist the live in-flight agent count to ``gateway_state.json`` at every turn boundary.
        Passes ONLY ``active_agents`` so the read-merge-write keeps lifecycle state (gateway_state=None
        would clobber it). Best-effort: a failed write must never disrupt a turn."""
        _write_runtime_status_quiet(active_agents=self._active_work_count())

    def _running_agent_ids(self) -> set:
        """``id()`` of every agent mid-turn — identity-keyed so the lookup is O(1) and independent of
        ``AIAgent.__eq__`` (MagicMock overrides it in tests)."""
        return {id(a) for _, a in self._running_agent_items()
                if a is not None and a is not _AGENT_PENDING_SENTINEL}

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        return {k: a for k, a in self._running_agent_items() if a is not _AGENT_PENDING_SENTINEL}

    # ---- Tunables consumed by the run_* mixins (kept on the class: tests and plugins patch them) ----

    # Per-session pending follow-up cap for busy_input_mode=queue (and paths sharing that entry point):
    # a stuck agent + rapid-fire user must not grow the overflow list unboundedly.
    _BUSY_QUEUE_MAX_PENDING = 32

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return
        # #28503 — Previously this called ``merge_pending_message_event``
        # with the default ``merge_text=False``, which silently OVERWROTE
        # the single pending slot when consecutive text messages arrived
        # in ``busy_input_mode: queue``. Route through the FIFO
        # infrastructure shared with ``/queue`` so each follow-up gets
        # its own turn in arrival order. Photo bursts still merge into
        # the head slot via ``merge_pending_message_event`` (album
        # semantics); everything else appends to the overflow tail.
        pending_slot = getattr(adapter, "_pending_messages", None)
        existing = pending_slot.get(session_key) if isinstance(pending_slot, dict) else None
        security_metadata_keys = (
            "hermes_plugin_id",
            "hermes_plugin_injection",
            "gateway_session_key",
            "gateway_session_id",
            "gateway_session_strict",
        )
        same_security_context = existing is not None and (
            getattr(existing, "internal", False) == getattr(event, "internal", False)
            and getattr(existing, "allow_gateway_control", True)
            == getattr(event, "allow_gateway_control", True)
            and all(
                (getattr(existing, "metadata", None) or {}).get(key)
                == (getattr(event, "metadata", None) or {}).get(key)
                for key in security_metadata_keys
            )
        )
        if same_security_context and (
            getattr(existing, "message_type", None) == MessageType.PHOTO
            or event.message_type == MessageType.PHOTO
            or bool(getattr(existing, "media_urls", None))
            or bool(getattr(event, "media_urls", None))
        ):
            # Preserve photo-burst / media-merge semantics for the head slot.
            merge_pending_message_event(
                adapter._pending_messages,
                session_key,
                event,
                merge_text=event.message_type == MessageType.TEXT,
            )
            return

        if self._queue_depth(session_key, adapter=adapter) >= self._BUSY_QUEUE_MAX_PENDING:
            logger.warning(
                "Dropping busy-mode follow-up for session %s — pending queue at cap (%d).",
                session_key,
                self._BUSY_QUEUE_MAX_PENDING,
            )
            return

        self._enqueue_fifo(session_key, event, adapter)

    async def _prepare_busy_steer_text(self, event: MessageEvent) -> str:
        """Return steerable text for a busy follow-up, transcribing voice first.

        Fresh and queued voice messages reach the normal inbound STT pipeline,
        but successful steer messages intentionally bypass that queue. Without
        preprocessing here, a media-only voice follow-up has an empty text
        payload and steer mode silently degrades to queue mode.

        Audio file attachments remain files; only voice-message media follows
        the automatic STT contract used by ``_prepare_inbound_message_text``.
        If transcription fails, preserve any caption and let the existing
        steer fallback handle an otherwise empty event without losing it.

        Routes through ``_transcribe_and_echo_pending_voice`` — the single
        out-of-band transcription choke point shared with the interrupt
        monitor and the pending-drain path — so the STT call is made at most
        once per platform message (cached on the event) and the transcript
        echo respects the count-based ledger.  If steering later falls back
        to queue mode, the drain path reuses the cached transcript instead of
        paying for a second STT call or re-echoing the same line.
        """
        text = (event.text or "").strip()
        if not self._pending_event_audio_paths(event):
            return text

        adapter = self._adapter_for_source(event.source)
        enriched_text, successful_transcripts = await self._transcribe_and_echo_pending_voice(
            event,
            adapter,
            event.source,
            text,
            log_context="Busy-steer",
        )
        if not successful_transcripts:
            return text
        return (enriched_text or text).strip()

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        # --- Authorization gate (#17775) ---
        # The cold path (_handle_message) checks _is_user_authorized before
        # creating a session.  The busy path must enforce the same check;
        # otherwise unauthorized users in shared threads (Slack/Telegram/Discord)
        # can inject messages into an active session they don't own.
        if not self._is_user_authorized(event.source):
            logger.warning(
                "Dropping message from unauthorized user in active session: "
                "user=%s (%s), platform=%s, session=%s",
                event.source.user_id,
                event.source.user_name,
                event.source.platform.value if event.source.platform else "unknown",
                session_key,
            )
            return True  # handled (silently dropped); do not fall through

        effective_mode = self._effective_busy_input_mode(event.source)

        # --- Draining case (gateway restarting/stopping) ---
        if self._draining:
            adapter = self._adapter_for_source(event.source)
            if not adapter:
                return True

            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
            if self._queue_during_drain_enabled(effective_mode):
                self._queue_or_replace_pending_event(session_key, event)
                message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
            else:
                message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."

            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
            return True

        # --- Approval response routing (#46866) ---
        # When the agent is blocked waiting for a dangerous-command approval,
        # plain-text responses like "yes" or "approve" must be routed to the
        # approval handler instead of being steered/queued/interrupted.
        # Otherwise approval via messaging platforms never succeeds — the
        # reply is queued behind a turn that can't start until the approval
        # resolves, so the approval times out and auto-denies (a deadlock).
        #
        # Slash forms (/approve, /deny) already bypass to the runner at the
        # base-adapter guard.  This handles the bare-word forms (Signal/SMS
        # users naturally type "yes" rather than "/approve").  Gating on
        # has_blocking_approval(session_key) is the disambiguator that keeps
        # a conversational "yes" from triggering a dangerous command when no
        # approval is actually pending (design intent — see run.py "Pending
        # exec approvals are handled by /approve and /deny" note).
        #
        # We reuse the canonical /approve and /deny handlers rather than
        # re-deriving the resolution + i18n messaging: they resolve the
        # waiting thread, resume typing, AND return a localized confirmation
        # string.  The busy-handler path does not auto-send that return, so
        # we deliver it ourselves (mirroring the draining-case send above).
        try:
            from tools.approval import has_blocking_approval
            if event.allow_gateway_control and has_blocking_approval(session_key):
                _raw_text = (event.text or "").strip().lower()
                _approve_words = {"approve", "yes", "ok", "okay", "confirm", "y", "👍"}
                _deny_words = {"deny", "no", "reject", "cancel", "n", "👎"}
                _approval_handler = None
                _normalized_args = ""
                if _raw_text in _approve_words:
                    _approval_handler = self._handle_approve_command
                elif _raw_text in _deny_words:
                    _approval_handler = self._handle_deny_command
                elif _raw_text in {"always", "approve always", "always approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "always"
                elif _raw_text in {"session", "approve session", "session approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "session"
                if _approval_handler is not None:
                    # Synthesize the canonical "/approve [args]" / "/deny"
                    # command text so the slash handlers parse modifiers via
                    # event.get_command_args().  Always use a literal "/" —
                    # MessageEvent.is_command()/get_command_args() only
                    # recognize the "/" prefix, not the per-platform display
                    # prefix ("!" on Slack/Matrix).
                    _verb = "approve" if _approval_handler is self._handle_approve_command else "deny"
                    _synth = f"/{_verb}"
                    if _normalized_args:
                        _synth = f"{_synth} {_normalized_args}"
                    event.text = _synth
                    _reply = await _approval_handler(event)
                    logger.info(
                        "Approval response via plain text: session=%s verb=%s args=%r",
                        session_key, _verb, _normalized_args,
                    )
                    _adapter = self._adapter_for_source(event.source)
                    if _adapter and _reply:
                        _text, _eph_ttl = _adapter._unwrap_ephemeral(_reply)
                        if _text:
                            _anchor = self._reply_anchor_for_event(event)
                            await _adapter._send_with_retry(
                                chat_id=event.source.chat_id,
                                content=_text,
                                reply_to=_anchor,
                                metadata=self._thread_metadata_for_source(event.source, _anchor),
                            )
                    return True
        except Exception:
            logger.warning(
                "Plain-text approval routing failed for session %s; "
                "falling through to busy handling",
                session_key, exc_info=True,
            )

        # Normal busy case (agent actively running a task)
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return False  # let default path handle it

        # --- Internal synthetic events must never interrupt/steer ---
        # Async-delegation completions (delegate_task(background=true)) and
        # background-process completions (terminal notify_on_complete) re-enter
        # the originating session as internal MessageEvents. When the session
        # is busy, treating them like a user TEXT message means interrupt-mode
        # (the default busy_text_mode) aborts the active turn AND sends a "⚡
        # Interrupting current task" ack — exactly the opposite of the design
        # invariant that a completion surfaces as a NEW turn only when idle and
        # never splices into a running turn. Plugin events carry untrusted
        # payload text, so queue those through the gateway FIFO to keep their
        # security metadata separate from pending user input.
        if getattr(event, "internal", False) and not event.allow_gateway_control:
            self._queue_or_replace_pending_event(session_key, event)
            return True
        if getattr(event, "internal", False):
            return False

        _busy_state = self._peek_session_state(session_key)
        running_agent = _busy_state.turn.agent if _busy_state else None

        # Discord's voice fast lane always queues a later utterance. The
        # global default is intentionally interrupt for typed correction
        # workflows, but speech often arrives in several STT chunks and an
        # interrupt here otherwise discards a nearly-complete spoken answer.
        # The FIFO retains MessageType.VOICE so the eventual reply still goes
        # through TTS. Explicit /stop and /new take their separate command
        # path before this handler and remain forceful controls.
        if self._is_voice_fast_lane_event(event):
            self._queue_or_replace_pending_event(session_key, event)
            logger.info("Queued voice fast-lane follow-up for busy session %s", session_key)
            return True

        busy_text_mode = self._effective_busy_text_mode(event.source)
        if (
            event.message_type == MessageType.TEXT
            and busy_text_mode == "queue"
            and effective_mode != "steer"
        ):
            return False

        # Steer mode: inject mid-run via running_agent.steer() instead of
        # queueing + interrupting.  If the agent isn't running yet
        # (sentinel) or lacks steer(), or the payload is empty, fall back
        # to queue semantics so nothing is lost.
        # #30170 — Subagent protection. ``AIAgent.interrupt()`` cascades
        # to every entry in the parent's ``_active_children`` list and
        # aborts in-flight ``delegate_task`` work. Demote ``interrupt``
        # to ``queue`` when the parent is currently driving subagents so
        # a conversational follow-up doesn't destroy minutes of subagent
        # work. Explicit ``/stop`` and ``/new`` slash commands go through
        # ``_interrupt_and_clear_session`` and are unaffected — the
        # operator still has a way to force-cancel everything.
        demoted_for_subagents = (
            effective_mode == "interrupt"
            and self._agent_has_active_subagents(running_agent)
        )
        if demoted_for_subagents:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because the running agent has active subagents (#30170)",
                session_key,
            )
            effective_mode = "queue"
        demoted_for_compression = (
            effective_mode == "interrupt"
            and await self._session_has_compression_in_flight(session_key)
        )
        if demoted_for_compression:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because context compression is in flight (#56391)",
                session_key,
            )
            effective_mode = "queue"
        steered = False
        redirected = False
        if effective_mode == "steer":
            steer_text = await self._prepare_busy_steer_text(event)
            # A follow-up qualifies for steering when it is plain text, OR
            # when every attachment is STT-eligible voice media whose
            # transcript was just folded into steer_text — otherwise a voice
            # note in steer mode silently degrades to queue mode (#58780).
            _steer_media_urls = getattr(event, "media_urls", None) or []
            _steer_all_voice = bool(_steer_media_urls) and (
                len(self._pending_event_audio_paths(event)) == len(_steer_media_urls)
            )
            can_steer = (
                steer_text
                and (
                    (
                        event.message_type == MessageType.TEXT
                        and not event.media_urls
                        and not event.media_types
                    )
                    or _steer_all_voice
                )
                and running_agent is not None
                and running_agent is not _AGENT_PENDING_SENTINEL
                and hasattr(running_agent, "steer")
            )
            if can_steer:
                try:
                    steered = bool(running_agent.steer(steer_text))
                except Exception as exc:
                    logger.warning("Gateway steer failed for session %s: %s", session_key, exc)
                    steered = False
            if not steered:
                # Fall back to queue (merge into pending messages, no interrupt)
                effective_mode = "queue"
        elif (
            effective_mode == "interrupt"
            and event.message_type == MessageType.TEXT
            and not event.media_urls
            and not event.media_types
            and running_agent is not None
            and running_agent is not _AGENT_PENDING_SENTINEL
            and getattr(running_agent, "_supports_active_turn_redirect", False) is True
            and hasattr(running_agent, "redirect")
        ):
            try:
                redirected = bool(running_agent.redirect((event.text or "").strip()))
            except Exception as exc:
                logger.warning("Gateway redirect failed for session %s: %s", session_key, exc)
                redirected = False

        # Store the message so it's processed as the next turn after the
        # current run finishes (or is interrupted).  Skip this for a
        # successful steer — the text already landed inside the run and
        # must NOT also be replayed as a next-turn user message.
        #
        # Route through _queue_or_replace_pending_event (the same FIFO
        # infrastructure used by busy queue-mode and /queue) rather than a
        # raw merge_pending_message_event(merge_text=True). The raw merge
        # newline-joins consecutive TEXT follow-ups into a SINGLE pending
        # turn, destroying message boundaries — so two separate user
        # messages sent while the agent was busy (interrupt mode, or a
        # steer that fell back to queue) arrived as one mashed-together
        # turn (#43066 sub-bug 2). The FIFO path gives each text its own
        # turn in arrival order while still preserving photo-burst / album
        # merge semantics for media.
        if not steered and not redirected:
            self._queue_or_replace_pending_event(session_key, event)

        is_queue_mode = effective_mode == "queue"
        is_steer_mode = effective_mode == "steer"
        is_redirect_mode = effective_mode == "interrupt" and redirected

        # If not in queue/steer mode, interrupt the running agent immediately.
        # This aborts in-flight tool calls and causes the agent loop to exit
        # at the next check point.
        if (
            effective_mode == "interrupt"
            and not redirected
            and running_agent
            and running_agent is not _AGENT_PENDING_SENTINEL
        ):
            try:
                _interrupt_text = event.text
                _media_urls = getattr(event, "media_urls", None) or []
                if self._pending_event_audio_paths(event):
                    _interrupt_text, _ = await self._transcribe_and_echo_pending_voice(
                        event,
                        adapter,
                        event.source,
                        event.text or "",
                        log_context="Voice-busy-interrupt",
                    )
                elif not _interrupt_text and _media_urls:
                    _interrupt_text = _build_media_placeholder(event)
                running_agent.interrupt(_interrupt_text)
            except Exception:
                pass  # don't let interrupt failure block the ack

        # Check if busy ack is disabled — skip sending but still process the input.
        # Placed before debounce so we don't stamp a "last ack" timestamp that was
        # never actually delivered.
        busy_ack_enabled = os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() == "true"
        if not busy_ack_enabled:
            logger.debug("Busy ack suppressed for session %s", session_key)
            return True  # input still processed, just no ack sent

        # Debounce before consulting config-heavy display settings. Rapid
        # follow-ups should be processed but should not trigger another config
        # read just to discover that no ack will be sent.
        _BUSY_ACK_COOLDOWN = 30
        now = time.time()
        last_ack = _busy_state.turn.busy_ack_ts if _busy_state else 0
        if now - last_ack < _BUSY_ACK_COOLDOWN:
            return True  # interrupt sent (if not queue), ack already delivered recently

        from gateway.display_config import resolve_display_setting
        platform_key = _platform_config_key(event.source.platform)

        # In steer mode the user's text has already been injected into the
        # active run. Some mobile chat setups want that steering to be silent,
        # like STT transcript echo suppression: keep the behavior, drop only
        # the confirmation bubble.
        if is_steer_mode:
            steer_ack_env = os.environ.get("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED")
            if steer_ack_env is not None:
                steer_ack_enabled = steer_ack_env.strip().lower() in {"1", "true", "yes", "on"}
            else:
                steer_ack_enabled = bool(
                    resolve_display_setting(
                        _load_gateway_config(),
                        platform_key,
                        "busy_steer_ack_enabled",
                        True,
                    )
                )
            if not steer_ack_enabled:
                logger.debug("Busy steer ack suppressed for session %s", session_key)
                return True

        self._session_state(session_key).turn.busy_ack_ts = now

        # Build a status-rich acknowledgment. Mobile chat defaults keep this
        # terse; detailed iteration/tool state is still available in logs and
        # can be opted in per platform via display.platforms.<platform>.busy_ack_detail.
        status_parts = []
        busy_ack_detail_enabled = bool(
            resolve_display_setting(
                _load_gateway_config(),
                _platform_config_key(event.source.platform),
                "busy_ack_detail",
                True,
            )
        )

        if busy_ack_detail_enabled and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                summary = running_agent.get_activity_summary()
                iteration = summary.get("api_call_count", 0)
                max_iter = summary.get("max_iterations", 0)
                current_tool = summary.get("current_tool")
                start_ts = _busy_state.turn.started_ts if _busy_state else 0
                if start_ts:
                    elapsed_min = int((now - start_ts) / 60)
                    if elapsed_min > 0:
                        status_parts.append(f"{elapsed_min} min elapsed")
                if max_iter:
                    status_parts.append(f"iteration {iteration}/{max_iter}")
                if current_tool:
                    status_parts.append(f"running: {current_tool}")
            except Exception:
                pass

        status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
        if is_steer_mode:
            message = (
                f"⏩ Steered into current run{status_detail}. "
                f"Your message arrives after the next tool call."
            )
        elif is_redirect_mode:
            message = (
                f"↪ Redirected current run{status_detail}. "
                f"I'll adjust using your correction."
            )
        elif is_queue_mode and demoted_for_subagents:
            # #30170 — explain the demotion so the user knows their
            # follow-up didn't accidentally kill the subagent and
            # discovers `/stop` as the explicit escape hatch.
            message = (
                f"⏳ Subagent working{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode and demoted_for_compression:
            message = (
                f"⏳ Compressing context{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode:
            message = (
                f"⏳ Queued for the next turn{status_detail}. "
                f"I'll respond once the current task finishes."
            )
        else:
            message = (
                f"⚡ Interrupting current task{status_detail}. "
                f"I'll respond to your message shortly."
            )

        # First-touch onboarding: the very first time a user sends a message
        # while the agent is busy, append a one-time hint explaining the
        # queue/interrupt knob.  Flag is persisted to config.yaml so it never
        # fires again on this install.
        try:
            from agent.onboarding import (
                BUSY_INPUT_FLAG,
                busy_input_hint_gateway,
                is_seen,
                mark_seen,
            )
            _user_cfg = _load_gateway_config()
            if not is_seen(_user_cfg, BUSY_INPUT_FLAG):
                if is_steer_mode:
                    _hint_mode = "steer"
                elif is_queue_mode:
                    _hint_mode = "queue"
                elif is_redirect_mode:
                    _hint_mode = "redirect"
                else:
                    _hint_mode = "interrupt"
                message = (
                    f"{message}\n\n"
                    f"{busy_input_hint_gateway(_hint_mode)}"
                )
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
        except Exception as _onb_err:
            logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)

        reply_anchor = self._reply_anchor_for_event(event)
        thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
        try:
            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
        except Exception as e:
            logger.debug("Failed to send busy-ack: %s", e)

        return True

    async def _drain_active_agents(
        self, timeout: float, cron_timeout: Optional[float] = None
    ) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_running_agents()
        last_active_count = self._running_agent_count()
        last_cron_count = self._active_cron_job_count()
        last_api_count = self._active_api_run_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_cron_count, last_api_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self._running_agent_count()
            cron_count = self._active_cron_job_count()
            api_count = self._active_api_run_count()
            if (
                force
                or active_count != last_active_count
                or cron_count != last_cron_count
                or api_count != last_api_count
                or (now - last_status_at) >= 1.0
            ):
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_cron_count = cron_count
                last_api_count = api_count
                last_status_at = now

        # Cron jobs run on the scheduler's own thread pool, outside
        # ``self._running_agents`` — fold their in-flight count into the
        # same wait/timeout this method already applies to chat sessions,
        # or a cron job's tool work gets killed with zero warning the
        # instant it's the only active thing running (#60432).
        # API-server / desk sessions have the same structural gap (#63529).
        if not self._running_agents and last_cron_count == 0 and last_api_count == 0:
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)

        # Cron work drains on its own deadline. ``timeout``
        # (``restart_drain_timeout``) defaults to 0 because interrupting a
        # chat turn is announced and resumable; a cron run killed mid-flight
        # is recorded in jobs.json as a permanent failure nobody is waiting
        # on. Sharing one budget meant the default config could report
        # ``timed_out=True`` after 0.00s with a cron job in flight and kill
        # it — the drain never even entered this loop (#82161).
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + timeout
        cron_deadline = started + (timeout if cron_timeout is None else cron_timeout)

        def _still_draining() -> bool:
            now = loop.time()
            if (
                len(self._running_agents) or self._active_api_run_count()
            ) and now < deadline:
                return True
            return bool(self._active_cron_job_count()) and now < cron_deadline

        # Both budgets at 0 leave this loop unentered, which is the legacy
        # "interrupt immediately" behaviour — expressed as an expired
        # deadline rather than a special case, so the timed_out value below
        # is always computed from real state instead of asserted up front.
        while _still_draining():
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = (
            bool(len(self._running_agents))
            or bool(self._active_cron_job_count())
            or bool(self._active_api_run_count())
        )
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                request_hard_interrupt(agent, reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)
        # API-server / desk turns are adapter-owned and never enter
        # _running_agents, so the loop above cannot see them even though
        # _drain_active_agents() waited for them (#63529).
        interrupted_api = self._interrupt_api_server_runs(reason)
        if interrupted_api:
            logger.debug("Interrupted %d api_server run(s) during shutdown", interrupted_api)

    async def _notify_interrupted_cron_jobs(self, job_ids) -> int:
        """Tell the owner of each just-interrupted cron job that its run died.

        The cron worker cannot do this itself. Its thread reaches
        ``_deliver_result`` asynchronously, and by then
        ``_bounded_adapter_teardown`` has closed the transport — so the notice
        never leaves the process, and ``_consume_interrupted_flag`` discards
        the resulting ``delivery_error`` along with it. The run's only trace is
        a line in jobs.json nobody reads (#82232).

        Must therefore be called from the post-interrupt phase, while adapters
        are still connected — the same window
        ``_notify_active_sessions_of_shutdown`` relies on for chat sessions,
        which is blind to cron work because cron runs on the scheduler's own
        thread pool rather than ``self._running_agents`` (#60432).

        Best-effort by construction: every failure is swallowed so a wedged
        adapter can never extend shutdown. Returns the number of notices sent.
        """
        if not job_ids:
            return 0
        try:
            from cron.jobs import get_job
            from cron.scheduler import _resolve_delivery_targets
        except Exception as e:
            logger.debug("Cron interrupt notification unavailable: %s", e)
            return 0

        action = "restarting" if self._restart_requested else "shutting down"
        notified: set = set()
        for job_id in job_ids:
            try:
                job = get_job(job_id)
                if not job:
                    continue
                # deliver=local jobs — and deliver=origin jobs with no
                # resolvable origin (#43014) — resolve to zero targets and
                # must stay silent rather than fall back to a home channel.
                targets = _resolve_delivery_targets(job)
            except Exception as e:
                logger.debug("Cron interrupt targets unresolved for %s: %s", job_id, e)
                continue
            if not targets:
                continue

            msg = (
                f"⚠️ Cron job '{job.get('name') or job_id}' was interrupted — "
                f"the gateway is {action} and killed the run before it "
                "finished. No result was produced for this run."
            )
            for target in targets:
                try:
                    platform = Platform(str(target.get("platform", "")).lower())
                except Exception:
                    continue
                adapter = self.adapters.get(platform)
                if adapter is None:
                    continue
                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    continue

                chat_id = str(target.get("chat_id"))
                thread_id = target.get("thread_id")
                dedup_key = (
                    job_id,
                    platform.value,
                    chat_id,
                    str(thread_id) if thread_id else None,
                )
                if dedup_key in notified:
                    continue
                try:
                    metadata = self._thread_metadata_for_target(
                        platform, chat_id, thread_id, adapter=adapter
                    )
                    result = await adapter.send(chat_id, msg, metadata=metadata)
                    if result is not None and getattr(result, "success", True) is False:
                        logger.debug(
                            "Cron interrupt notice to %s:%s failed: %s",
                            platform.value, chat_id,
                            getattr(result, "error", "send returned success=False"),
                        )
                        continue
                    notified.add(dedup_key)
                except Exception as e:
                    logger.debug(
                        "Cron interrupt notice to %s:%s raised: %s",
                        platform.value, chat_id, e,
                    )
        if notified:
            logger.info(
                "Shutdown: delivered %d interrupted-cron-job notice(s)",
                len(notified),
            )
        return len(notified)

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the very start of stop() — adapters are still connected so
        messages can be delivered. Best-effort: individual send failures are
        logged and swallowed so they never block the shutdown sequence.
        """
        active = self._snapshot_running_agents()
        restart_source = self._restart_command_source if self._restart_requested else None

        action = "restarting" if self._restart_requested else "shutting down"
        hint = (
            "Your current task will be interrupted. "
            "Send any message after restart and I'll try to resume where you left off."
            if self._restart_requested
            else "Your current task will be interrupted."
        )
        msg = f"⚠️ Gateway {action} — {hint}"

        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in active:
            source = None
            try:
                if getattr(self, "session_store", None) is not None:
                    await self.async_session_store._ensure_loaded()
                    entry = self.session_store._entries.get(session_key)
                    source = getattr(entry, "origin", None) if entry else None
            except Exception as e:
                logger.debug(
                    "Failed to load session origin for shutdown notification %s: %s",
                    session_key,
                    e,
                )

            if source is None:
                source = self._get_cached_session_source(session_key)

            if source is not None:
                platform_str = source.platform.value
                chat_id = str(source.chat_id)
                thread_id = source.thread_id
            else:
                # Fall back to parsing the session key when no persisted
                # origin is available (legacy sessions/tests).
                _parsed = _parse_session_key(session_key)
                if not _parsed:
                    continue
                platform_str = _parsed["platform"]
                chat_id = _parsed["chat_id"]
                thread_id = _parsed.get("thread_id")

            # Deduplicate only identical delivery targets. Thread/topic-aware
            # platforms can share a parent chat while still routing to distinct
            # destinations via metadata.
            dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
            if dedup_key in notified:
                continue

            try:
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue

                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    logger.info(
                        "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                        platform_str,
                    )
                    continue

                reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
                if reply_to_message_id is None and restart_source is not None:
                    try:
                        restart_platform = restart_source.platform.value
                        restart_chat_id = str(restart_source.chat_id)
                        restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                        if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                            reply_to_message_id = getattr(restart_source, "message_id", None)
                    except Exception:
                        pass

                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=getattr(source, "chat_type", None) if source is not None else None,
                    reply_to_message_id=reply_to_message_id,
                    adapter=adapter,
                )

                result = await adapter.send(chat_id, msg, metadata=metadata)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to %s:%s: %s",
                        platform_str,
                        chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to active chat %s:%s",
                    platform_str, chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str, chat_id, e,
                )

        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return

        # Suppress ONLY the home-channel broadcast when the drain that is ending
        # in this shutdown asked us to be quiet (e.g. a NAS auto-update image
        # migration — drain-gated, then the machine is recreated). On the
        # always-on Hermes Cloud fleet that broadcast would otherwise fire on
        # every routine auto-update, spamming home channels with operator-
        # flavoured "gateway shutting down" pings the user doesn't care about.
        # The per-active-session interrupt pings above are deliberately NOT
        # gated: on a drained shutdown they're empty by construction, and in the
        # force-interrupt (deadline-exceeded) case they carry the genuinely
        # useful "your task was cut off, message me to resume" hint. The flag is
        # only honoured for a CURRENT-epoch marker (drain_notification_suppressed
        # reuses the NS-570 staleness check), so an orphaned marker can never
        # silence a fresh gateway's legitimate broadcast.
        try:
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                logger.info(
                    "Home-channel shutdown broadcast suppressed by drain marker "
                    "(suppress_notification=true)"
                )
                return
        except Exception as e:
            # Never let the suppression check block the shutdown broadcast —
            # fail toward the louder, more-visible behaviour.
            logger.debug("drain_notification_suppressed check failed: %s", e)

        # Snapshot adapters up front: adapter.send() can hit a fatal error
        # path that pops the adapter from self.adapters (see _handle_fatal
        # elsewhere), which would otherwise trigger
        # ``RuntimeError: dictionary changed size during iteration`` —
        # observed in a user report during gateway shutdown.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if dedup_key in notified:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
                else:
                    result = await adapter.send(str(home.chat_id), msg)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to home channel %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to home channel %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    e,
                )

    async def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            # Persist any in-flight transcript to the SQLite session store
            # before teardown (#13121).  An agent forcibly interrupted by the
            # drain-timeout escalation may never reach
            # ``turn_finalizer.finalize_turn`` (the only place that flushes the
            # turn to state.db) — e.g. it was blocked in a tool call that did
            # not abort within the post-interrupt grace window.  Its in-flight
            # tool rounds live only in the in-memory ``_session_messages``
            # (refreshed per tool round in ``conversation_loop`` but never
            # written to SQLite mid-turn), so the immediate pre-restart turn is
            # silently dropped from ``load_transcript()`` on resume.  Flushing
            # here closes that gap; the resume_pending / fresh-tool-tail
            # branches in ``_handle_message_with_agent`` already expect a
            # transcript whose tail may be a pending tool result.  The flush is
            # idempotent (identity-tracked in ``_flush_messages_to_session_db``),
            # so agents that DID finish gracefully re-flush nothing.
            try:
                _flush = getattr(agent, "_flush_messages_to_session_db", None)
                _session_messages = getattr(agent, "_session_messages", None)
                if callable(_flush) and isinstance(_session_messages, list) and _session_messages:
                    # Strip private empty-response retry scaffolding from the
                    # tail first, mirroring the graceful ``_persist_session``
                    # path, so a resumed turn doesn't replay synthetic recovery
                    # nudges.
                    _strip = getattr(
                        agent, "_drop_trailing_empty_response_scaffolding", None
                    )
                    if callable(_strip):
                        try:
                            _strip(_session_messages)
                        except Exception:
                            pass
                    try:
                        _flush(_session_messages)
                    except Exception as _flush_err:
                        # The in-memory transcript could not be persisted
                        # (e.g. FTS/SQLite index corruption — #72680). A plain
                        # debug log loses the conversation permanently when the
                        # process exits. Dump the live agent history to an
                        # external JSON recovery snapshot so an operator can
                        # salvage it after repairing state.db. The flush is
                        # non-fatal; shutdown must never block on a best-effort
                        # backup.
                        logger.warning(
                            "Shutdown transcript flush failed (%s); preserving "
                            "%d in-memory message(s) to recovery snapshot",
                            _flush_err,
                            len(_session_messages),
                        )
                        from gateway.shutdown_flush import flush_agent_history_to_file
                        flush_agent_history_to_file(
                            getattr(agent, "session_id", None),
                            _session_messages,
                        )
            except Exception as _e:
                logger.debug("Shutdown transcript flush failed: %s", _e)
            # Off-loop + bounded: finalize_session fans out to plugin
            # on_session_finalize hooks that can do arbitrary synchronous
            # work (e.g. an observability plugin serializing a full-session
            # trace export). Running it inline on the event loop blocked the
            # entire shutdown sequence past systemd's TimeoutStopSec on a
            # multi-day 4.7G session — heartbeats froze and the process was
            # SIGKILLed mid-export. Same class as the memory-provider hang
            # below (#53175).
            await self._finalize_session_off_loop(
                session_id=getattr(agent, "session_id", None),
                platform="gateway",
                reason="shutdown",
            )
            # Off-loop + bounded: a wedged memory provider here used to hang
            # the whole shutdown so SIGTERM never completed (#53175).
            await self._cleanup_agent_resources_off_loop(
                agent, context="shutdown finalize"
            )

    def _should_emit_long_running_notification(
        self,
        session_key: Optional[str],
        agent: Any,
        executor_task: Optional[Any],
    ) -> bool:
        """Only emit the heartbeat while this task still owns the live run.

        Guards against a stale ``running: delegate_task`` heartbeat outliving the
        run that started it: stop once the executor finishes, the agent is gone,
        or the session key has been rebound to a different live agent (e.g. the
        user sent ``/new`` and a fresh agent took the slot mid-run, #12029).
        """
        if agent is None:
            return False
        if executor_task is not None and executor_task.done():
            return False
        if session_key:
            _hb_state = self._peek_session_state(session_key)
            if (_hb_state.turn.agent if _hb_state else None) is not agent:
                return False
        return True

    # Upper bound on off-loop agent-resource cleanup invoked from coroutines
    # running on the gateway's event loop (session-expiry sweep, in-turn
    # cache-hygiene re-eviction). _cleanup_agent_resources is synchronous and
    # can block for a long time (agent.close() does subprocess teardown;
    # shutdown_memory_provider() may do network/SQLite IO via a memory plugin).
    # Calling it inline wedges the whole loop — the bot goes silent, the
    # runtime-status updated_at heartbeat freezes, and SIGTERM cannot be
    # serviced (#53175). Offload to a worker thread under this timeout so the
    # loop is never blocked; mirrors the /new reset path's fix (#35994).
    @dataclasses.dataclass
    class _BusySteerOutcome:
        effective_mode: str
        demoted_for_subagents: bool
        demoted_for_compression: bool
        steered: bool
        redirected: bool

    # Worker bound for _cleanup_agent_resources: sync, can block long (subprocess teardown, memory IO).
    _CLEANUP_TIMEOUT_S = 30.0

    # Budget for one finalize_session() dispatch (plugin on_session_finalize hooks + Relay close):
    # enough for a normal trace-export flush, small enough a wedged plugin can't eat the stop window.
    _FINALIZE_TIMEOUT_S = 10.0

    _STUCK_LOOP_THRESHOLD = 3  # restarts while active before auto-suspend
    _STUCK_LOOP_FILE = ".restart_failure_counts"

    # Reasons set by _stop_impl() on force-interrupt; "restart_interrupted" by suspend_recently_active()
    # on crash recovery (no .clean_shutdown marker). All mean "killed mid-turn" -> startup auto-resume.
    _AUTO_RESUME_REASONS = frozenset({"restart_timeout", "shutdown_timeout", "restart_interrupted"})

    _MAX_SUPERVISED_RESTARTS = 5
    # Ran this long before crashing = HEALTHY (isolated crash, not a crash-loop); restart counter resets.
    _SUPERVISED_HEALTHY_SECS = 300
    # Slow respawn tier once the watcher's restart budget is spent; long on purpose (crashes on contact).
    _RECONNECT_WATCHER_SLOW_RETRY_SECS = 300
    # Slow-tier respawns while work is queued; if 30 min of 5-min retries can't keep it up, fail loudly.
    _MAX_SLOW_WATCHER_RESPAWNS = 6
    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0
    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes
    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling
    # Command-specific mid-run reject texts (busy_policy == "reject" with a busy_handler naming an
    # entry here); all other rejected commands get the generic text in _dispatch_busy_slash_command.
    _BUSY_REJECT_TEXT: Dict[str, str] = {
        "model": "Agent is running — wait or /stop first, then switch models.",
        "codex-runtime": "Agent is running — wait or /stop first, then change runtime.",
        "moa": "Agent is running — wait or /stop first, then run /moa."}

    def _active_profile_name(self) -> str:
        """Return the profile name this gateway represents."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    def _is_user_authorized_for_source(
        self, source: SessionSource, *, allow_adapter_delegation: bool = True) -> bool:
        """Authorize under the live transport's profile, not the routed runtime (which need not copy the
        shared bot token/allowlist); the transport home is stamped on the source for this read only."""
        def _check() -> bool:
            # Keep the one-argument seam used by plugins/tests; pass the keyword only when disabling.
            if allow_adapter_delegation:
                return self._is_user_authorized(source)
            return self._is_user_authorized(source, allow_adapter_delegation=False)

        return self._under_authorization_profile(source, _check)

    def _admit_bot_message_for_source(self, source: SessionSource) -> bool:
        """Count a bot message under the profile that authorized it, so the guard's peek, count and
        config all read the transport profile's ``gateway.bot_loop_guard``."""
        return self._under_authorization_profile(source, lambda: self._admit_bot_message(source))

    @staticmethod
    def _under_authorization_profile(source: SessionSource, check):
        authorization_home = getattr(source, "_authorization_profile_home", None)
        if authorization_home is None:
            return check()
        with _profile_runtime_scope(Path(authorization_home)):
            return check()

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    @property
    def async_session_store(self) -> AsyncSessionStore:
        """Return the single async facade for this runner's SessionStore."""
        facade = getattr(self, "_async_session_store", None)
        if facade is None or facade._store is not self.session_store:
            facade = AsyncSessionStore(self.session_store)
            self._async_session_store = facade
        return facade

    def _get_cached_session_source(self, session_key: str):
        cached_sources = getattr(self, "_session_sources", None) if session_key else None
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            with suppress(Exception):
                cached_sources.move_to_end(session_key)
        return source

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _msg_preview = (event.text or "")[:80].replace("\n", " ")
        _reply_id = getattr(event, "reply_to_message_id", None)
        _reply_txt = (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " ")
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", _msg_preview, _reply_id, _reply_txt,
        )

        # Get or create session
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's
        # last-active topic so a cross-topic Reply or stripped plain reply
        # doesn't fragment the conversation across sessions.
        recovered = await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            try:
                event.source = source
            except Exception:
                pass

        event_metadata = getattr(event, "metadata", None) or {}
        expected_session_key = str(
            event_metadata.get("gateway_session_key") or ""
        ).strip()
        if expected_session_key:
            derived_session_key = self._session_key_for_source(source)
            if derived_session_key != expected_session_key:
                logger.warning(
                    "Dropping internally routed event after route recovery: "
                    "expected session=%s derived=%s",
                    expected_session_key,
                    derived_session_key,
                )
                return

        strict_session = bool(event_metadata.get("gateway_session_strict"))
        pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
        if strict_session:
            session_entry = await self.async_session_store.lookup_by_session_key(
                expected_session_key
            )
            if (
                session_entry is None
                or not pinned_session_id
                or session_entry.session_id != pinned_session_id
            ):
                logger.warning(
                    "Dropping internally routed event: expected session id=%s is no "
                    "longer current for key=%s",
                    pinned_session_id or "missing",
                    expected_session_key or "missing",
                )
                return
        else:
            # Internal wakes must observe reset policy without becoming user
            # activity themselves. Otherwise periodic Kanban/process
            # notifications keep the stable routing key alive across every
            # daily/idle boundary.
            session_entry = await self.async_session_store.get_or_create_session(
                source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        session_key = session_entry.session_key
        if not strict_session and pinned_session_id:
            resolved_entry = await self._resolve_async_delegation_session(
                session_entry,
                pinned_session_id,
            )
            if resolved_entry is None:
                return
            session_entry = resolved_entry
        self._cache_session_source(session_key, source)
        if await asyncio.to_thread(self._is_telegram_topic_lane, source):
            try:
                binding = (await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )) if self._session_db else None
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                bound_session_id = str(binding.get("session_id") or "")
                # Heal bindings that point at a pre-compression parent: walk
                # the compression-continuation chain forward to its tip so the
                # next message resumes the compressed child instead of
                # reloading the oversized parent transcript (#20470/#29712/
                # #33414). Returns the input unchanged when the session isn't
                # a compression parent, so this is cheap and safe.
                if bound_session_id and self._session_db is not None:
                    try:
                        canonical_session_id = await self._session_db.get_compression_tip(
                            bound_session_id,
                        )
                    except Exception:
                        logger.debug(
                            "compression-tip lookup failed for %s",
                            bound_session_id, exc_info=True,
                        )
                        canonical_session_id = bound_session_id
                    if (
                        canonical_session_id
                        and canonical_session_id != bound_session_id
                    ):
                        bound_session_id = canonical_session_id
                if bound_session_id and bound_session_id != session_entry.session_id:
                    # Route the override through SessionStore so the session_key
                    # → session_id mapping is persisted to disk and the previous
                    # lane session is ended cleanly. Mutating session_entry in
                    # place here created a split-brain state where the JSON
                    # index pointed at one id but code downstream used another.
                    switched = await self.async_session_store.switch_session(session_key, bound_session_id)
                    if switched is not None:
                        session_entry = switched
                # If the stored binding pointed at a parent, rewrite it to the
                # canonical descendant now that we've followed the chain.
                if (
                    bound_session_id
                    and bound_session_id != str(binding.get("session_id") or "")
                ):
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-tip-walk",
                    )
            else:
                try:
                    await asyncio.to_thread(self._record_telegram_topic_binding, source, session_entry)
                except Exception:
                    logger.debug("Failed to record Telegram topic binding", exc_info=True)
        # Capture and immediately consume was_auto_reset so it does not
        # re-fire on subsequent messages — preventing the cleanup from
        # wiping model/reasoning overrides set between turns (Closes #48031).
        _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
        if _was_auto_reset:
            # Treat auto-reset as a full conversation boundary — clear every
            # conversation-scoped per-session dict in one funnel call so the
            # fresh session does not inherit the previous conversation's
            # model/reasoning overrides, a queued "/model switched" note, or
            # a stale resolved-model cache (#48031, #58403). See
            # _CONVERSATION_SCOPED_STATE.
            self._clear_conversation_scope(session_key, reason="auto_reset")
            # Evict the cached agent so the fresh session does not inherit the
            # previous conversation's context_compressor._previous_summary —
            # the cache is keyed on the stable session_key, so an auto-reset
            # otherwise reuses the old agent and leaks prior history into new
            # compaction summaries. Mirrors /reset and the compression-exhausted
            # path (#9893). Covers daily/idle/suspended auto-reset.
            self._evict_cached_agent(session_key)
            session_entry.was_auto_reset = False

        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or _was_auto_reset
            or getattr(session_entry, "is_fresh_reset", False)
        )
        # Consume the is_fresh_reset flag immediately so it doesn't leak
        # onto subsequent messages in the same session (issue #6508).
        if getattr(session_entry, "is_fresh_reset", False):
            session_entry.is_fresh_reset = False
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })

        # Build session context
        context = build_session_context(source, self.config, session_entry)

        # Set session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)

        # Read privacy.redact_pii from config (re-read per message)
        _redact_pii = False
        persist_user_message = None
        persist_user_timestamp = None
        # Synthetic self-injected turns (async-delegation batch completions,
        # background watch notifications, resume wake-ups) arrive as
        # MessageEvent(internal=True). Persist their user row typed with
        # display_kind="internal_notification" so transcripts/UIs can render
        # them as timeline notices instead of user bubbles (#82888). Role and
        # content are untouched — display_kind is a DB-only sidecar stripped
        # from every provider-bound payload (see conversation_loop's
        # api_msg.pop("display_kind")).
        persist_user_display_kind = (
            "internal_notification" if getattr(event, "internal", False) else None
        )
        try:
            _pcfg = _load_gateway_config()
            _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # Build the context prompt to inject.  The render is pinned per
        # session, keyed by a hash of the exact renderer inputs
        # (_ephemeral_change_key).  A key hit reuses the pinned bytes verbatim
        # so the composed system prompt cannot drift turn-over-turn; a key
        # miss (thread rename, /sethome, redact_pii flip, ...) re-renders
        # once — the only legitimate cache busts.
        context_prompt = self._pinned_session_context_prompt(
            context, _redact_pii, session_key
        )

        # Per-turn must-deliver notes.  These used to be appended to
        # context_prompt (the ephemeral system prompt), which guaranteed a
        # turn1→turn2 system-prompt diff and a full agent rebuild.  They now
        # ride the current user message via the api_content sidecar instead
        # (staged below, consumed in run_sync → build_turn_context).
        turn_sidecar_notes: List[str] = []

        # If the previous session expired and was auto-reset, deliver a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if _was_auto_reset:
            reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
            if reset_reason == "suspended":
                context_note = "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]"
            elif reset_reason == "daily":
                context_note = "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]"
            elif reset_reason == "resume_pending_expired":
                context_note = "[System note: The previous gateway session could not be recovered after a restart (API recovery timed out). This is a fresh conversation — use /resume to restore history if needed.]"
            else:
                context_note = "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]"
            # Slack/Discord channels/threads are long-lived: point the agent at
            # the specific prior same-channel session so it recalls that context
            # via session_search instead of an unrelated recent session.  Returns
            # None (appends nothing) for other platforms or when there's no prior
            # activity to recall.  Deterministic — no extra API/DB calls (#36220).
            try:
                continuity_note = build_channel_continuity_note(session_entry, source)
            except Exception:
                continuity_note = None
            if continuity_note:
                context_note = context_note + "\n\n" + continuity_note
            turn_sidecar_notes.append(context_note)

            # Send a user-facing notification explaining the reset, unless:
            # - notifications are disabled in config
            # - the platform is excluded (e.g. api_server, webhook)
            # - the expired session had no activity (nothing was cleared)
            try:
                policy = self.session_store.config.get_reset_policy(
                    platform=source.platform,
                    session_type=getattr(source, 'chat_type', 'dm'),
                )
                platform_name = source.platform.value if source.platform else ""
                had_activity = getattr(session_entry, 'reset_had_activity', False)
                # Suspended and restart-recovery-expired sessions always notify
                # regardless of policy.notify — the user had an active session
                # that was silently replaced, so they need to know they can
                # /resume it.  Idle/daily resets respect the policy flag.
                should_notify = reset_reason in {"suspended", "resume_pending_expired"} or (
                    policy.notify
                    and had_activity
                    and platform_name not in policy.notify_exclude_platforms
                )
                if should_notify:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        if reset_reason == "suspended":
                            reason_text = "previous session was stopped or interrupted"
                        elif reset_reason == "resume_pending_expired":
                            reason_text = "gateway restart recovery timed out"
                        elif reset_reason == "daily":
                            reason_text = f"daily schedule at {policy.at_hour}:00"
                        else:
                            hours = policy.idle_minutes // 60
                            mins = policy.idle_minutes % 60
                            duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
                            reason_text = f"inactive for {duration}"
                        notice = (
                            f"◐ Session automatically reset ({reason_text}). "
                            f"Conversation history cleared.\n"
                            f"Use /resume to browse and restore a previous session.\n"
                            f"Adjust reset timing in config.yaml under session_reset."
                        )
                        try:
                            session_info = await asyncio.to_thread(
                                self._reset_notice_session_info, source
                            )
                            if session_info:
                                notice = f"{notice}\n\n{session_info}"
                        except Exception:
                            pass
                        await adapter.send(
                            source.chat_id, notice,
                            metadata=self._thread_metadata_for_source(source),
                        )
            except Exception as e:
                logger.debug("Auto-reset notification failed (non-fatal): %s", e)

            # was_auto_reset is already consumed in the cleanup block above
            # (single source of truth); only the reset reason needs clearing here.
            session_entry.auto_reset_reason = None

        # Auto-load skill(s) for topic/channel bindings (Telegram DM Topics,
        # Discord channel_skill_bindings).  Supports a single name or ordered list.
        # Only inject on NEW sessions — ongoing conversations already have the
        # skill content in their conversation history from the first message.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
            try:
                from agent.skill_commands import _load_skill_payload, _build_skill_message
                _combined_parts: list[str] = []
                _loaded_names: list[str] = []
                for _sname in _skill_names:
                    _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                    if _loaded:
                        _loaded_skill, _skill_dir, _display_name = _loaded
                        _note = (
                            f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                            f"Follow its instructions for this session.]"
                        )
                        _part = _build_skill_message(_loaded_skill, _skill_dir, _note)
                        if _part:
                            _combined_parts.append(_part)
                            _loaded_names.append(_sname)
                    else:
                        logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                if _combined_parts:
                    # Append the user's original text after all skill payloads
                    _combined_parts.append(event.text)
                    event.text = "\n\n".join(_combined_parts)
                    logger.info(
                        "[Gateway] Auto-loaded skill(s) %s for session %s",
                        _loaded_names, session_key,
                    )
            except Exception as e:
                logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

        # ── Turn lease (#64934) ────────────────────────────────────────
        # Session resolution is FINAL here (get_or_create → async-delegation
        # pinning → topic tip-walk switch_session are all above). Serialize
        # the [load history → run → flush] region per resolved SESSION_ID:
        # when a second routing key is mapped to this same session_id, its
        # turn waits here for the previous turn's flush instead of loading a
        # stale history base and interleaving transcript writes. Same-key
        # messages never reach this point mid-turn (adapter + runner guards
        # hold them), so the lock is uncontended outside the alias-key route.
        # Fail-closed on timeout: never enter the transcript region without a
        # lease. Outer dispatch returns a bounded rejection/resend notice rather
        # than recreating the exact concurrent-turn corruption this lease exists
        # to prevent. Released in _handle_message's finally via
        # _release_turn_lease — granted per (routing key, run generation) so a
        # stale unwind can't release a newer turn's lease.
        _lease_registry = getattr(self, "_turn_leases", None)
        if _lease_registry is not None:
            try:
                _lease_token = await _lease_registry.acquire(
                    session_entry.session_id,
                    owner_key=_quick_key,
                    generation=run_generation,
                    timeout=_float_env(
                        "HERMES_TURN_LEASE_TIMEOUT", DEFAULT_LEASE_WAIT
                    ),
                )
            except TurnLeaseTimeoutError:
                # The broad session-context cleanup finally starts later in this
                # method. Restore the tokens here before propagating the rejection
                # to outer dispatch, or this early exit leaks task-local identity.
                self._clear_session_env(_session_env_tokens)
                raise
            if _lease_token is not None:
                _lease_state = self._session_state(_quick_key).turn
                _lease_state.lease_token = _lease_token
                _lease_state.lease_generation = run_generation

        # A turn only becomes durable recovery work after it owns (or has
        # explicitly degraded past) the per-session lease.  Marking before the
        # await above would falsely recover an alias-routed message that never
        # began processing if the gateway died while it was still waiting.
        await self._mark_durable_active_turn(event, session_entry.session_key)

        # Load conversation history from transcript
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        # -----------------------------------------------------------------
        # Session hygiene: auto-compress pathologically large transcripts
        #
        # Long-lived gateway sessions can accumulate enough history that
        # every new message rehydrates an oversized transcript, causing
        # repeated truncation/context failures.  Detect this early and
        # compress proactively — before the agent even starts.  (#628)
        #
        # Token source priority:
        # 1. Actual API-reported prompt_tokens from the last turn
        #    (stored in session_entry.last_prompt_tokens)
        # 2. Rough char-based estimate (str(msg)//4). Overestimates
        #    by 30-50% on code/JSON-heavy sessions, but that just
        #    means hygiene fires a bit early — safe and harmless.
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length_async,
            )

            # Read model + compression config from config.yaml.
            # NOTE: hygiene threshold is intentionally HIGHER than the agent's
            # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
            # sessions that grew too large between turns — it fires pre-agent
            # to prevent API failures.  The agent's own compressor handles
            # normal context management during its tool loop with accurate
            # real token counts.  Having hygiene at 0.50 caused premature
            # compression on every turn in long gateway sessions.
            _hyg_model = "anthropic/claude-sonnet-4.6"
            _hyg_threshold_pct = 0.85
            _hyg_compression_enabled = True
            _hyg_hard_msg_limit = 5000
            _hyg_timeout_seconds = 30.0
            _hyg_total_ceiling_seconds = 600.0
            _hyg_failure_cooldown_seconds = 300.0
            _hyg_config_context_length = None
            _hyg_provider = None
            _hyg_base_url = None
            _hyg_api_key = None
            _hyg_configured_model = None
            _hyg_configured_provider = None
            _hyg_configured_base_url = None
            _hyg_data = {}
            try:
                _hyg_data = _load_gateway_config()
                if _hyg_data:
                    # Resolve model name (same logic as run_sync)
                    _model_cfg = _hyg_data.get("model", {})
                    if isinstance(_model_cfg, str):
                        _hyg_model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        _hyg_model = _model_cfg.get("default") or _model_cfg.get("model") or _hyg_model
                        # Read explicit context_length override from model config
                        # (same as run_agent.py lines 995-1005)
                        _raw_ctx = _model_cfg.get("context_length")
                        if _raw_ctx is not None:
                            try:
                                _hyg_config_context_length = int(_raw_ctx)
                            except (TypeError, ValueError):
                                pass
                        # Read provider for accurate context detection
                        _hyg_provider = _model_cfg.get("provider") or None
                        _hyg_base_url = _model_cfg.get("base_url") or None

                    # Read compression settings — only use enabled flag.
                    # The threshold is intentionally separate from the agent's
                    # compression.threshold (hygiene runs higher).
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in {"true", "1", "yes"}
                        _raw_hard_limit = _comp_cfg.get("hygiene_hard_message_limit")
                        if _raw_hard_limit is not None:
                            try:
                                _parsed = int(_raw_hard_limit)
                                if _parsed > 0:
                                    _hyg_hard_msg_limit = _parsed
                            except (TypeError, ValueError):
                                pass
                        _raw_timeout = _comp_cfg.get("hygiene_timeout_seconds")
                        if _raw_timeout is not None:
                            try:
                                _parsed = float(_raw_timeout)
                                if _parsed > 0:
                                    _hyg_timeout_seconds = _parsed
                            except (TypeError, ValueError):
                                pass
                        _raw_ceiling = _comp_cfg.get("hygiene_total_ceiling_seconds")
                        if _raw_ceiling is not None:
                            try:
                                _parsed = float(_raw_ceiling)
                                if _parsed > 0:
                                    _hyg_total_ceiling_seconds = _parsed
                            except (TypeError, ValueError):
                                pass
                        # The ceiling can never be tighter than one idle
                        # window, or the extension loop would be dead code.
                        _hyg_total_ceiling_seconds = max(
                            _hyg_total_ceiling_seconds, _hyg_timeout_seconds,
                        )
                        _raw_cooldown = _comp_cfg.get("hygiene_failure_cooldown_seconds")
                        if _raw_cooldown is not None:
                            try:
                                _parsed = float(_raw_cooldown)
                                if _parsed >= 0:
                                    _hyg_failure_cooldown_seconds = _parsed
                            except (TypeError, ValueError):
                                pass

                _hyg_configured_model = _hyg_model
                _hyg_configured_provider = _hyg_provider
                _hyg_configured_base_url = _hyg_base_url

                try:
                    _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                    )
                    _hyg_provider = _hyg_runtime.get("provider") or _hyg_provider
                    _hyg_base_url = _hyg_runtime.get("base_url") or _hyg_base_url
                    _hyg_api_key = _hyg_runtime.get("api_key") or _hyg_api_key
                except Exception:
                    pass

                if _hyg_config_context_length is not None:
                    try:
                        from hermes_cli.route_identity import should_clear_context_pin_async

                        if await should_clear_context_pin_async(
                            _hyg_configured_model,
                            _hyg_model,
                            _hyg_configured_base_url,
                            _hyg_base_url,
                            _hyg_configured_provider,
                            _hyg_provider,
                        ):
                            _hyg_config_context_length = None
                    except Exception:
                        _hyg_config_context_length = None

                # Check custom_providers per-model context_length
                # (same fallback as run_agent.py lines 1171-1189).
                # Must run after runtime resolution so _hyg_base_url is set.
                if _hyg_config_context_length is None and _hyg_base_url:
                    try:
                        try:
                            from hermes_cli.config import (
                                get_compatible_custom_providers as _gw_gcp,
                                get_custom_provider_context_length as _gw_gccl,
                            )
                            _hyg_custom_providers = _gw_gcp(_hyg_data)
                        except Exception:
                            _hyg_custom_providers = _hyg_data.get("custom_providers")
                            if not isinstance(_hyg_custom_providers, list):
                                _hyg_custom_providers = []
                        _hyg_custom_ctx = _gw_gccl(
                            model=_hyg_model,
                            base_url=_hyg_base_url,
                            custom_providers=_hyg_custom_providers,
                        )
                        if _hyg_custom_ctx:
                            _hyg_config_context_length = int(_hyg_custom_ctx)
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

            if _hyg_compression_enabled:
                _hyg_context_length = await get_model_context_length_async(
                    _hyg_model,
                    base_url=_hyg_base_url or "",
                    api_key=_hyg_api_key or "",
                    config_context_length=_hyg_config_context_length,
                    provider=_hyg_provider or "",
                )
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)

                # Prefer actual API-reported tokens from the last turn
                # (stored in session entry) over the rough char-based estimate.
                _stored_tokens = session_entry.last_prompt_tokens
                if _stored_tokens > 0:
                    _approx_tokens = _stored_tokens
                    _token_source = "actual"
                else:
                    _approx_tokens = estimate_messages_tokens_rough(history)
                    _token_source = "estimated"
                    # Note: rough estimates overestimate by 30-50% for code/JSON-heavy
                    # sessions, but that just means hygiene fires a bit early — which
                    # is safe and harmless.  The 85% threshold already provides ample
                    # headroom (agent's own compressor runs at 50%).  A previous 1.4x
                    # multiplier tried to compensate by inflating the threshold, but
                    # 85% * 1.4 = 119% of context — which exceeds the model's limit
                    # and prevented hygiene from ever firing for ~200K models (GLM-5).

                # Hard safety valve: force compression if message count is
                # extreme, regardless of token estimates.  This breaks the
                # death spiral where API disconnects prevent token data
                # collection, which prevents compression, which causes more
                # disconnects.  5000 messages is far above any normal session
                # but catches truly runaway growth before it becomes
                # unrecoverable.  Set well clear of legitimate large-context
                # (1M+) sessions doing thousands of short turns — those
                # compress on the token threshold, not this count-based floor.
                # Threshold is configurable via
                # compression.hygiene_hard_message_limit.
                # (#2153)
                _HARD_MSG_LIMIT = _hyg_hard_msg_limit
                _needs_compress = (
                    _approx_tokens >= _compress_token_threshold
                    or _msg_count >= _HARD_MSG_LIMIT
                )

                if _needs_compress:
                    # Use the persistent DB-backed cooldown (same as the
                    # in-conversation compression path in context_compressor.py)
                    # so the cooldown survives gateway restarts. The in-memory
                    # dict was reset on every restart, re-triggering the same
                    # failing compression and wedging session storage (#74136).
                    _session_db = getattr(self, "_session_db", None)
                    if _session_db is not None:
                        _session_db = getattr(_session_db, "_db", _session_db)
                        _getter = getattr(_session_db, "get_compression_failure_cooldown", None)
                        if _getter is not None:
                            try:
                                _cooldown_state = _getter(session_entry.session_id)
                            except Exception:
                                _cooldown_state = None
                            if _cooldown_state and _cooldown_state.get("remaining_seconds", 0) > 0:
                                logger.info(
                                    "Session hygiene: skipping compression for %s; "
                                    "previous failure cooldown active for %.1fs",
                                    session_entry.session_id,
                                    _cooldown_state["remaining_seconds"],
                                )
                                _needs_compress = False

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}", _token_source,
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

                    try:
                        from agent.conversation_compression import CompressionCommitFence
                        from run_agent import AIAgent

                        _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                            source=source,
                            session_key=session_key,
                            user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                        )
                        if _hyg_runtime.get("api_key"):
                            # Pass the FULL transcript (tool results included).
                            # Filtering to user/assistant-only starved the
                            # compressor: tool results are usually the bulk of
                            # the context, _prune_old_tool_results never saw
                            # them, and short filtered histories tripped the
                            # protect-first/last early-return so nothing was
                            # compressed at all (#3854). The agent loop passes
                            # its full message list to _compress_context — the
                            # gateway now matches.
                            _hyg_msgs = [
                                m for m in history
                                if m.get("role") in {"user", "assistant", "tool"}
                            ]

                            if len(_hyg_msgs) >= 4:
                                try:
                                    _hyg_session_row = await self._session_db.get_session(
                                        session_entry.session_id
                                    )
                                except Exception as exc:
                                    _hyg_session_row = None
                                    logger.warning(
                                        "Session hygiene could not restore the system "
                                        "prompt for session %s: %s. Preserving an empty "
                                        "prompt so the live turn rebuilds it with its "
                                        "configured providers.",
                                        session_entry.session_id,
                                        exc,
                                        exc_info=True,
                                    )
                                _hyg_session_db = getattr(self._session_db, "_db", self._session_db)
                                # Hygiene performs the same lossy rewrite as
                                # normal compression. When the operator enabled
                                # compression.checkpoint_required, the memory
                                # provider must be loaded so the required
                                # checkpoint is created before any transcript
                                # mutation; otherwise keep the historical fast
                                # path (no provider init, no best-effort hook)
                                # for hygiene.
                                from hermes_cli.config import load_config as _load_cfg
                                from utils import is_truthy_value as _is_truthy

                                _hyg_checkpoint_required = _is_truthy(
                                    ((_load_cfg() or {}).get("compression") or {}).get(
                                        "checkpoint_required"
                                    ),
                                    default=False,
                                )
                                _hyg_agent = AIAgent(
                                    **_hyg_runtime,
                                    model=_hyg_model,
                                    max_iterations=4,
                                    quiet_mode=True,
                                    skip_memory=not _hyg_checkpoint_required,
                                    enabled_toolsets=["memory"],
                                    session_id=session_entry.session_id,
                                    session_db=_hyg_session_db,
                                )
                                _seed_hygiene_system_prompt(
                                    _hyg_agent,
                                    _hyg_session_row,
                                )
                                # If compression must rebuild instead of retaining
                                # the cached prompt, make the persisted result
                                # deliberately stale for every real gateway surface.
                                _hyg_agent.platform = _GATEWAY_HYGIENE_PLATFORM
                                _hyg_cleanup_deferred = False
                                try:
                                    # Gateway hygiene runs before the user turn
                                    # starts and already owns the session binding.
                                    # Prefer in-place compaction here: it archives
                                    # old rows under the same session id instead of
                                    # minting a continuation child that then has to
                                    # be published back to SessionStore/topic
                                    # bindings.  If no SessionDB is available,
                                    # compress_context leaves this flag false and
                                    # the guard below preserves the transcript.
                                    _hyg_agent.compression_in_place = True
                                    _bind_hyg_state = getattr(
                                        getattr(_hyg_agent, "context_compressor", None),
                                        "bind_session_state",
                                        None,
                                    )
                                    if callable(_bind_hyg_state):
                                        _bind_hyg_state(
                                            _hyg_session_db,
                                            session_entry.session_id,
                                        )
                                    # It must never finalize on close() — close()
                                    # would end the live gateway session row.
                                    _hyg_agent._end_session_on_close = False
                                    _hyg_agent._print_fn = lambda *a, **kw: None

                                    loop = asyncio.get_running_loop()
                                    _hyg_commit_fence = CompressionCommitFence()
                                    _hyg_future = loop.run_in_executor(
                                        None,
                                        lambda: _hyg_agent._compress_context(
                                            _hyg_msgs, "",
                                            approx_tokens=_approx_tokens,
                                            commit_fence=_hyg_commit_fence,
                                        ),
                                    )
                                    try:
                                        # Progress-aware wait: the timeout is an
                                        # INACTIVITY budget, not a total one. The
                                        # compression worker streams its summary
                                        # call and ticks the fence per token
                                        # (CompressionCommitFence.touch_progress),
                                        # so a slow reasoning model that is still
                                        # generating keeps extending the deadline;
                                        # only a genuinely silent worker times out.
                                        # A hard ceiling bounds the total wait so
                                        # a degenerate trickle stream can't hold
                                        # the turn forever.
                                        _hyg_wait_started = time.monotonic()
                                        while True:
                                            # #76354 S3: charge the idle budget
                                            # from the LAST PROGRESS event, not
                                            # from the start of this wait slice —
                                            # otherwise silence can approach 2x
                                            # the configured timeout.
                                            _slice = max(
                                                _hyg_timeout_seconds
                                                - _hyg_commit_fence.seconds_since_progress(),
                                                0.005,
                                            )
                                            try:
                                                _compressed, _ = await asyncio.wait_for(
                                                    asyncio.shield(_hyg_future),
                                                    timeout=_slice,
                                                )
                                                break
                                            except asyncio.TimeoutError:
                                                _hyg_waited = time.monotonic() - _hyg_wait_started
                                                _idle = _hyg_commit_fence.seconds_since_progress()
                                                if (
                                                    _idle < _hyg_timeout_seconds
                                                    and _hyg_waited < _hyg_total_ceiling_seconds
                                                ):
                                                    logger.info(
                                                        "Session hygiene compression for "
                                                        "session %s still streaming after "
                                                        "%.0fs (last progress %.1fs ago) — "
                                                        "extending wait (ceiling %.0fs)",
                                                        session_entry.session_id,
                                                        _hyg_waited, _idle,
                                                        _hyg_total_ceiling_seconds,
                                                    )
                                                    continue
                                                raise
                                    except asyncio.TimeoutError:
                                        _cancelled = None
                                        while _cancelled is None:
                                            # #76354 F1: a hung commit retains the
                                            # fence lock; the lock-free phase
                                            # marker keeps this loop from spinning
                                            # forever while the commit blocks.
                                            if _hyg_commit_fence.commit_in_flight:
                                                _cancelled = False
                                                break
                                            _cancelled = (
                                                _hyg_commit_fence.try_cancel_before_commit()
                                            )
                                            if _cancelled is None:
                                                # Round-2 #5: transient
                                                # lock-setup windows ride
                                                # write patience for seconds;
                                                # 25ms keeps sub-tick latency
                                                # without 1kHz spin.
                                                await asyncio.sleep(0.025)
                                        if not _cancelled:
                                            # The worker crossed the commit boundary just
                                            # before the timeout. The fence poll waited for
                                            # that boundary to finish, so consume the
                                            # completed result instead of treating a
                                            # successful compaction as a timeout.
                                            _compressed, _ = await _hyg_future
                                        else:
                                            # #76354 F4: release the timed-out
                                            # worker's durable lease via the
                                            # holder-qualified hook so the next
                                            # compressor can acquire the lock
                                            # immediately (no ABA against a new
                                            # holder — release is holder-scoped).
                                            _hyg_commit_fence.release_cancelled_compression_lock()
                                            self._defer_agent_cleanup_until_future_done(
                                                _hyg_future,
                                                _hyg_agent,
                                                context="session hygiene timeout",
                                            )
                                            _hyg_cleanup_deferred = True
                                            if _hyg_failure_cooldown_seconds >= 0:
                                                _hyg_cooldown = await asyncio.to_thread(
                                                    _hygiene_cooldown_for_failure,
                                                    self,
                                                    session_key,
                                                    _hyg_failure_cooldown_seconds,
                                                )
                                                _record_hygiene_cooldown(
                                                    self, session_entry.session_id,
                                                    _hyg_cooldown,
                                                    "session hygiene compression "
                                                    "timed out with no output from "
                                                    "the summary model",
                                                )
                                            from agent.session_activity import (
                                                ActivityProvenance,
                                            )
                                            _stamp_hygiene_compression_provenance(
                                                _hyg_agent,
                                                "session hygiene compression timed out",
                                                ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                                                "hygiene compression timeout "
                                                "activity stamp failed",
                                            )
                                            logger.warning(
                                                "Session hygiene compression for session %s "
                                                "made no progress for %.1fs "
                                                "(total wait %.1fs, ceiling %.1fs); "
                                                "continuing without compression",
                                                session_entry.session_id,
                                                _hyg_commit_fence.seconds_since_progress(),
                                                time.monotonic() - _hyg_wait_started,
                                                _hyg_total_ceiling_seconds,
                                            )
                                            _timeout_msg = (
                                                "⚠️ Context compression timed out "
                                                f"after {_hyg_timeout_seconds:.1f}s "
                                                "with no output from the summary model. "
                                                "No messages were dropped — continuing without "
                                                "compression. Run /compress to retry, /reset for "
                                                "a clean session, or check your "
                                                "auxiliary.compression model configuration."
                                            )
                                            try:
                                                _adapter = self._adapter_for_source(source)
                                                if _adapter and source.chat_id:
                                                    await _adapter.send(
                                                        source.chat_id,
                                                        _timeout_msg,
                                                        metadata=_hyg_meta,
                                                    )
                                            except Exception as _werr:
                                                logger.warning(
                                                    "Failed to deliver compression-timeout "
                                                    "warning to user: %s",
                                                    _werr,
                                                )
                                            raise
                                    except BaseException:
                                        # #76354 F2: non-timeout unwind while the
                                        # detached hygiene worker may still run —
                                        # KeyboardInterrupt, task cancellation, or
                                        # any unexpected error. Revoke commit
                                        # admission (and release the worker's
                                        # durable lease via the holder-qualified
                                        # hook) BEFORE the host unwinds so the
                                        # worker can never commit later.
                                        _hyg_commit_fence.revoke_commit_admission()
                                        if not _hyg_cleanup_deferred:
                                            self._defer_agent_cleanup_until_future_done(
                                                _hyg_future,
                                                _hyg_agent,
                                                context="session hygiene unwind",
                                            )
                                            _hyg_cleanup_deferred = True
                                        raise

                                    # _compress_context ends the old session and creates
                                    # a new session_id.  Write compressed messages into
                                    # the NEW session so the old transcript stays intact
                                    # and searchable via session_search.
                                    _hyg_new_sid = _hyg_agent.session_id
                                    _hyg_rotated = _hyg_new_sid != session_entry.session_id
                                    _hyg_in_place = bool(
                                        getattr(_hyg_agent, "_last_compaction_in_place", False)
                                    )
                                    # Anti-growth guard: refuse a compression
                                    # that did not shrink the transcript
                                    # (observed: 427K -> 598K). Compare
                                    # like-for-like rough estimates.
                                    _hyg_in_toks = estimate_messages_tokens_rough(history)
                                    _hyg_out_toks = estimate_messages_tokens_rough(_compressed)
                                    if _hyg_rotated and _hyg_out_toks > _hyg_in_toks:
                                        logger.warning(
                                            "Gateway hygiene compression for session %s "
                                            "would grow transcript (~%s -> ~%s tokens); "
                                            "keeping the original transcript unchanged",
                                            session_entry.session_id,
                                            f"{_hyg_in_toks:,}",
                                            f"{_hyg_out_toks:,}",
                                        )
                                        _hyg_rotated = False
                                        _compressed = history
                                    # Only rewrite the transcript when rotation produced
                                    # a NEW session id.  In-place compaction does NOT
                                    # need a rewrite: archive_and_compact() has already
                                    # soft-archived the previous active rows and inserted
                                    # the compacted messages as the new active set inside
                                    # _compress_context().  Calling rewrite_transcript()
                                    # after in-place compaction would invoke
                                    # replace_messages(active_only=False) which DELETEs
                                    # ALL rows — including the archived turns that
                                    # archive_and_compact() deliberately preserved
                                    # (silent data loss, #61145).
                                    #
                                    # The danger this guards against (mirrors the
                                    # /compress fix #44794/#39704): if _compress_context
                                    # returns a summary but neither rotates nor completes
                                    # archive_and_compact(), the session_id is unchanged
                                    # for a FAILURE reason, and an unconditional
                                    # rewrite_transcript() would DELETE the original
                                    # messages and replace them with only the compressed
                                    # summary (permanent data loss, #21301).
                                    #
                                    # Write-before-repoint (mirrors manual /compress):
                                    # if we repointed session_entry onto the child SID
                                    # and rewrite_transcript then failed (lock/ENOSPC),
                                    # the live entry would already reference a brand-new
                                    # empty session while the turn continues — the
                                    # conversation silently vanishes. Persist the child
                                    # transcript first; only then rebind the live entry.
                                    if _hyg_rotated:
                                        if not await self.async_session_store.rewrite_transcript(
                                            _hyg_new_sid, _compressed
                                        ):
                                            logger.error(
                                                "Session hygiene: failed to persist "
                                                "compressed transcript for rotated "
                                                "session %s → %s; keeping the live "
                                                "entry on the original session so the "
                                                "conversation is not dropped",
                                                session_entry.session_id,
                                                _hyg_new_sid,
                                            )
                                            # Fail closed: treat like no rotation.
                                            _hyg_rotated = False
                                            _hyg_in_place = False
                                        else:
                                            session_entry.session_id = _hyg_new_sid
                                            # The held turn lease follows the
                                            # rotation so an alias key resolving
                                            # the fresh child still serializes
                                            # against this turn (#64934).
                                            self._rebind_turn_lease(
                                                _quick_key, run_generation, _hyg_new_sid
                                            )
                                            await self.async_session_store._save()
                                            await asyncio.to_thread(
                                                self._sync_telegram_topic_binding,
                                                source, session_entry,
                                                reason="hygiene-compression",
                                            )

                                    if _hyg_rotated:
                                        # Reset stored token count — transcript rewritten
                                        session_entry.last_prompt_tokens = 0
                                        history = _compressed
                                        _new_count = len(_compressed)
                                        _new_tokens = estimate_messages_tokens_rough(
                                            _compressed
                                        )
                                    elif _hyg_in_place:
                                        # archive_and_compact() already persisted the
                                        # compacted transcript inside _compress_context.
                                        # Reset counts to match the new active set.
                                        session_entry.last_prompt_tokens = 0
                                        history = _compressed
                                        _new_count = len(_compressed)
                                        _new_tokens = estimate_messages_tokens_rough(
                                            _compressed
                                        )
                                    else:
                                        # No rewrite happened — transcript preserved
                                        # unchanged, so the post-compression counts equal
                                        # the pre-compression ones.
                                        _new_count = _msg_count
                                        _new_tokens = _approx_tokens
                                        logger.warning(
                                            "Gateway hygiene compression for session %s "
                                            "did not rotate or compact in place "
                                            "(no session_db on the hygiene agent) — "
                                            "preserving the original transcript instead "
                                            "of overwriting it with the summary (#21301).",
                                            session_entry.session_id,
                                        )

                                    logger.info(
                                        "Session hygiene: compressed %s → %s msgs, "
                                        "~%s → ~%s tokens",
                                        _msg_count, _new_count,
                                        f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                    )

                                    if _new_tokens >= _warn_token_threshold:
                                        logger.warning(
                                            "Session hygiene: still ~%s tokens after "
                                            "compression",
                                            f"{_new_tokens:,}",
                                        )

                                    # If summary generation failed, the
                                    # compressor aborts entirely and returns
                                    # messages unchanged — nothing is dropped.
                                    # Surface a visible warning to the gateway
                                    # user — agent.log alone is invisible on
                                    # TG/Discord/etc. — so they know the chat
                                    # is "frozen" at the current size and can
                                    # /compress to retry or /reset to start
                                    # fresh.
                                    _comp = getattr(_hyg_agent, "context_compressor", None)
                                    _hyg_aborted = _comp is not None and getattr(
                                        _comp, "_last_compress_aborted", False
                                    )
                                    if not _hyg_aborted:
                                        # Recovery decision lives in the
                                        # extracted, unit-tested predicate — the
                                        # degenerate "did not rotate or compact
                                        # in place" path (#21301) sets both flags
                                        # False and reuses the pre-compression
                                        # counts, so a numbers-only check would
                                        # read a no-op as success and clear the
                                        # streak on every wedged run (#79624).
                                        if hygiene_compaction_recovered(
                                            aborted=_hyg_aborted,
                                            rotated=_hyg_rotated,
                                            in_place=_hyg_in_place,
                                            msg_count=_msg_count,
                                            new_count=_new_count,
                                            approx_tokens=_approx_tokens,
                                            new_tokens=_new_tokens,
                                        ):
                                            await asyncio.to_thread(
                                                _reset_hygiene_failure_streak,
                                                self,
                                                session_key,
                                            )
                                    if _hyg_aborted:
                                        if _hyg_failure_cooldown_seconds >= 0:
                                            _hyg_cooldown = await asyncio.to_thread(
                                                _hygiene_cooldown_for_failure,
                                                self,
                                                session_key,
                                                _hyg_failure_cooldown_seconds,
                                            )
                                            _record_hygiene_cooldown(
                                                self, session_entry.session_id,
                                                _hyg_cooldown,
                                                getattr(
                                                    _comp, "_last_summary_error", None
                                                ),
                                            )
                                        from agent.session_activity import (
                                            ActivityProvenance,
                                        )
                                        _stamp_hygiene_compression_provenance(
                                            _hyg_agent,
                                            "session hygiene compression aborted",
                                            ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
                                            "hygiene compression abort "
                                            "activity stamp failed",
                                        )
                                        _err = getattr(_comp, "_last_summary_error", None) or "unknown error"
                                        # Force-redact: provider exception text
                                        # may contain credentials; this message
                                        # reaches gateway users directly.
                                        from agent.redact import redact_sensitive_text
                                        _err = redact_sensitive_text(_err, force=True)
                                        _warn_msg = (
                                            "⚠️ Context compression aborted "
                                            f"({_err}). No messages were dropped — "
                                            "conversation is unchanged. Run /compress "
                                            "to retry, /reset for a clean session, or "
                                            "check your auxiliary.compression model "
                                            "configuration."
                                        )
                                        try:
                                            _adapter = self._adapter_for_source(source)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _warn_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver compression-failure warning to user: %s",
                                                _werr,
                                            )
                                    # Separately: if the user's CONFIGURED aux
                                    # model failed and we recovered by falling
                                    # back to the main model, tell them — a
                                    # misconfigured auxiliary.compression.model
                                    # is something only they can fix, and
                                    # silent recovery would hide it.
                                    elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
                                        _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
                                        _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
                                        _aux_msg = (
                                            f"ℹ️ Configured compression model `{_aux_model}` "
                                            f"failed ({_aux_err}). Recovered using your main "
                                            "model — context is intact — but you may want to "
                                            "check `auxiliary.compression.model` in config.yaml."
                                        )
                                        try:
                                            _adapter = self._adapter_for_source(source)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _aux_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver aux-model-fallback notice to user: %s",
                                                _werr,
                                            )
                                finally:
                                    # Evict the cached agent so the next turn
                                    # rebuilds its system prompt from current
                                    # SOUL.md, memory, and skills.
                                    self._evict_cached_agent(session_key)
                                    if not _hyg_cleanup_deferred:
                                        await self._cleanup_agent_resources_off_loop(
                                            _hyg_agent, context="session hygiene"
                                        )

                    except Exception as e:
                        logger.warning(
                            "Session hygiene auto-compress failed: %s", e
                        )

        # First-message onboarding -- only on the very first interaction ever.
        # Delivered on the current user message (sidecar), NOT the ephemeral
        # system prompt: present-on-turn-1/absent-on-turn-2 was a guaranteed
        # system-prompt diff and agent rebuild.
        if not history and not await self.async_session_store.has_any_sessions():
            # Default first-contact note: a brief self-introduction.
            _intro_note = (
                "[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
            # Opt-in structured profile-build path. When enabled (default
            # "ask") and not yet offered on this install, swap the plain intro
            # for a consent-gated directive that offers to build a user
            # profile and persists confirmed facts via memory(target="user").
            # The offer fires at most once (onboarding.seen flag); set
            # onboarding.profile_build: off in config.yaml to disable.
            try:
                from agent.onboarding import (
                    PROFILE_BUILD_FLAG,
                    is_seen,
                    mark_seen,
                    profile_build_directive,
                    profile_build_mode,
                )
                _onb_cfg = _load_gateway_config()
                if (
                    profile_build_mode(_onb_cfg) == "ask"
                    and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG)
                ):
                    turn_sidecar_notes.append(profile_build_directive().strip())
                    mark_seen(_hermes_home / "config.yaml", PROFILE_BUILD_FLAG)
                else:
                    turn_sidecar_notes.append(_intro_note)
            except Exception as _pb_err:
                logger.debug(
                    "Profile-build onboarding directive failed, using plain intro: %s",
                    _pb_err,
                )
                turn_sidecar_notes.append(_intro_note)

        # One-time prompt if no home channel is set for this platform
        # Skip for webhooks - they deliver directly to configured targets (github_comment, etc.)
        if not history and source.platform and source.platform != Platform.LOCAL and source.platform != Platform.WEBHOOK:
            platform_name = source.platform.value
            env_key = _home_target_env_var(platform_name)
            # Multiplex: home channel may live only in the profile secret
            # scope / PlatformConfig, not process os.environ.
            home_env = ""
            try:
                from agent.secret_scope import get_secret

                home_env = (get_secret(env_key) or "").strip() if env_key else ""
            except Exception:
                home_env = ""
            if not home_env:
                home_env = (os.getenv(env_key) or "").strip() if env_key else ""
            # Also honor in-memory / yaml home_channel on this platform.
            try:
                if not home_env and self.config.get_home_channel(source.platform):
                    home_env = "set"
            except Exception:
                pass
            # Secondary-profile platforms (e.g. Slack on yolo) may only exist
            # under that profile's loaded config — check after scope install.
            if not home_env:
                try:
                    from gateway.config import load_gateway_config as _lgc
                    prof = (getattr(source, "profile", None) or "").strip()
                    if prof and prof != "default":
                        # Already inside profile scope for secondary handlers;
                        # re-read live config for home_channel.
                        _pcfg = _lgc()
                        if _pcfg.get_home_channel(source.platform):
                            home_env = "set"
                except Exception:
                    pass
            if not home_env:
                # Slack dispatches all Hermes commands through a single
                # parent slash command `/hermes`; bare `/sethome` is not
                # registered and would fail with "app did not respond".
                sethome_cmd = (
                    "/hermes sethome"
                    if source.platform == Platform.SLACK
                    else "/sethome"
                )
                notice = (
                    f"📬 No home channel is set for {platform_name.title()}. "
                    f"A home channel is where Hermes delivers cron job results "
                    f"and cross-platform messages.\n\n"
                    f"Type {sethome_cmd} to make this chat your home channel, "
                    f"or ignore to skip."
                )
                await self._deliver_platform_notice(source, notice)

        # -----------------------------------------------------------------
        # Voice channel awareness — deliver current voice channel state so
        # the agent knows who is in the channel and who is speaking, without
        # needing a separate tool call.  Delivered on the current user
        # message and ONLY when it changed since the previous turn: the
        # member/speaking serialization differs essentially every turn, and
        # appending it to the ephemeral system prompt forced a full agent
        # rebuild + prompt-cache re-key per message.  The system prompt
        # carries a static pointer line instead (gateway/session.py).
        # -----------------------------------------------------------------
        _vc_note = self._voice_channel_sidecar_note(event, source, session_key)
        if _vc_note:
            turn_sidecar_notes.append(_vc_note)

        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        message_text = await self._prepare_profile_scoped_inbound_message_text(
            event=event,
            source=source,
            history=history,
            session_key=session_key,
        )
        if message_text is None:
            return

        # Capture the platform event time as message metadata and keep the
        # persisted transcript clean (strip any leading timestamp prefix).
        # This runs regardless of the toggle so storage stays clean and the
        # send-time is preserved. Only the in-context RENDER (prepending the
        # human-readable prefix the model sees) is gated behind
        # gateway.message_timestamps.enabled — default OFF.
        try:
            from hermes_time import get_timezone as _get_evt_tz
            from gateway.message_timestamps import (
                coerce_message_timestamp as _coerce_msg_ts,
                render_user_content_with_timestamp as _render_msg_ts,
                strip_leading_message_timestamps as _strip_msg_ts,
            )
            _evt_tz = _get_evt_tz()
            _evt_ts = getattr(event, "timestamp", None)
            if message_text and isinstance(message_text, str):
                _clean_message_text, _embedded_ts = _strip_msg_ts(
                    message_text, tz=_evt_tz)
                persist_user_message = _clean_message_text
                _event_epoch = _coerce_msg_ts(_evt_ts, tz=_evt_tz)
                persist_user_timestamp = (
                    _event_epoch if _event_epoch is not None else _embedded_ts
                )
                if _message_timestamps_enabled(_load_gateway_config()):
                    message_text = _render_msg_ts(
                        _clean_message_text,
                        persist_user_timestamp,
                        tz=_evt_tz,
                    )
                else:
                    # Toggle off: model sees the clean message; the timestamp
                    # is still stored as metadata for later opt-in.
                    message_text = _clean_message_text
        except Exception as _ts_err:
            logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)

        # Stage the collected must-deliver notes for this turn's agent run
        # (one-shot; consumed in run_sync).  Staged AFTER the message_text
        # early-out above so an aborted turn cannot leak its notes into the
        # next turn's user message.
        if turn_sidecar_notes and session_key:
            self._set_pending_turn_sidecar_notes(session_key, turn_sidecar_notes)

        # Bind this gateway run generation to the adapter's active-session
        # event so deferred post-delivery callbacks can be released by the
        # same run that registered them.
        self._bind_adapter_run_generation(
            self._adapter_for_source(source),
            session_key,
            run_generation,
        )

        try:
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(getattr(source, "thread_id", None)) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Run the agent. Capture the session id that this run was launched
            # against so post-run compression publication can be identity-guarded
            # below; a /new or another lifecycle transition may move
            # session_entry.session_id while the old run is still unwinding.
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=_run_start_session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=self._reply_anchor_for_event(event),
                channel_prompt=event.channel_prompt,
                moa_config=getattr(event, "_moa_config", None),
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=event.message_type,
            )
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            # Stop persistent typing indicator now that the agent is done.
            # Slack AI status is scoped to a thread/workspace, so preserve the
            # same routing metadata used by the response delivery path.
            try:
                _typing_adapter = self._adapter_for_source(source)
                _stop_with_metadata = getattr(
                    type(_typing_adapter), "_stop_typing_with_metadata", None
                )
                _stop_typing = getattr(type(_typing_adapter), "stop_typing", None)
                if _typing_adapter and callable(_stop_with_metadata):
                    await _typing_adapter._stop_typing_with_metadata(
                        source.chat_id,
                        self._thread_metadata_for_source(
                            source, self._reply_anchor_for_event(event)
                        ),
                    )
                elif _typing_adapter and callable(_stop_typing):
                    await _typing_adapter.stop_typing(source.chat_id)
            except Exception:
                pass

            if not self._is_session_run_current(_quick_key, run_generation):
                logger.info(
                    "Discarding stale agent result for %s — generation %d is no longer current",
                    _quick_key or "?",
                    run_generation,
                )
                _stale_adapter = self._adapter_for_source(source)
                if getattr(type(_stale_adapter), "pop_post_delivery_callback", None) is not None:
                    _stale_adapter.pop_post_delivery_callback(
                        _quick_key,
                        generation=run_generation,
                    )
                elif _stale_adapter and hasattr(_stale_adapter, "_post_delivery_callbacks"):
                    _stale_adapter._post_delivery_callbacks.pop(_quick_key, None)
                return None

            response = agent_result.get("final_response") or ""
            # Hidden-reasoning-only retry exhaustion: the loop's sentinel text
            # ("Codex response remained incomplete after 3 continuation
            # attempts") doubles as final_response, so it would be delivered
            # verbatim into the channel — where peer agents can ingest it as a
            # completed assistant turn (#51628). Blank it here so the normal
            # empty-response handling (and the suppression below) applies.
            if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
                response = ""
            try:
                from gateway.response_filters import is_intentional_silence_agent_result
                _intentional_silence = is_intentional_silence_agent_result(
                    agent_result, response,
                )
            except Exception:
                _intentional_silence = False

            # Convert the agent's internal "(empty)" sentinel into a
            # user-friendly message.  "(empty)" means the model failed to
            # produce visible content after exhausting all retries (nudge,
            # prefill, empty-retry, fallback).  Sending the raw sentinel
            # looks like a bug; a short explanation is more helpful.
            if response == "(empty)" and not _intentional_silence:
                response = (
                    "⚠️ The model returned no response after processing tool "
                    "results. This can happen with some models — try again or "
                    "rephrase your question."
                )
            agent_messages = agent_result.get("messages", [])
            _response_time = time.time() - _msg_start_time
            _api_calls = agent_result.get("api_calls", 0)
            _resp_len = len(response)
            logger.info(
                "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
                _platform_name, source.chat_id or "unknown",
                _response_time, _api_calls, _resp_len,
            )

            # NOTE: the cross-process cache-coherence re-baseline
            # (_refresh_agent_cache_message_count) is intentionally deferred
            # until AFTER this turn's transcript persistence block below — it
            # must include the first-turn `session_meta` marker row and the
            # compression session_id swap, both of which happen later.  See
            # the call site after the `update_session(...)` write.

            # Successful turn — clear any stuck-loop counter for this session.
            # This ensures the counter only accumulates across CONSECUTIVE
            # restarts where the session was active (never completed).
            #
            # Also clear the resume_pending flag (set by drain-timeout
            # shutdown) — the turn ran to completion, so recovery
            # succeeded and subsequent messages should no longer receive
            # the restart-interruption system note.
            if session_key and _should_clear_resume_pending_after_turn(agent_result):
                await self._clear_restart_failure_count(session_key)
                try:
                    await self.async_session_store.clear_resume_pending(session_key)
                except Exception as _e:
                    logger.debug(
                        "clear_resume_pending failed for %s: %s",
                        session_key, _e,
                    )

            # Normalize empty responses: surface errors, partial failures, and
            # the case where agent did work but returned no text. Fix for #18765.
            if not _intentional_silence:
                response = _normalize_empty_agent_response(
                    agent_result, response, history_len=len(history),
                )
                response = _sanitize_gateway_final_response(source.platform, response)

            # Ordering contract: the agent thread already updated the contextvar
            # in conversation_compression.py; propagate to SessionEntry + _save().
            # If the agent's session_id changed during compression, update
            # session_entry so transcript writes below go to the right session.
            if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
                if session_entry.session_id == _run_start_session_id:
                    session_entry.session_id = agent_result["session_id"]
                    # The held turn lease follows the rotation: the transcript
                    # persistence below writes to the NEW id, so the
                    # serialization boundary must move with it or an alias
                    # key resolving the fresh child could interleave (#64934).
                    self._rebind_turn_lease(
                        _quick_key, run_generation, session_entry.session_id
                    )
                    await self.async_session_store._save()
                    await self.async_session_store._record_gateway_session_peer(
                        session_entry.session_id,
                        session_key,
                        source,
                    )
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="agent-result-compression",
                    )
                else:
                    logger.info(
                        "Skipping agent-result session split sync for %s because "
                        "the session binding moved from %s to %s before "
                        "compression finished",
                        session_key or "?",
                        _run_start_session_id,
                        session_entry.session_id,
                    )

            # Prepend reasoning/thinking if display is enabled (per-platform).
            # Mattermost requires explicit per-platform opt-in because this is
            # scratch text, not ordinary final-answer content.
            try:
                _show_reasoning_effective = _resolve_gateway_display_bool(
                    _load_gateway_config(),
                    _platform_config_key(source.platform),
                    "show_reasoning",
                    default=bool(getattr(self, "_show_reasoning", False)),
                    platform=source.platform,
                    require_platform_override_for={Platform.MATTERMOST},
                )
            except Exception:
                _show_reasoning_effective = (
                    False
                    if source.platform == Platform.MATTERMOST
                    else getattr(self, "_show_reasoning", False)
                )
            if _show_reasoning_effective and response and not _intentional_silence:
                last_reasoning = agent_result.get("last_reasoning")
                if last_reasoning:
                    from gateway.stream_consumer import escape_code_fences_for_display
                    # Collapse long reasoning to keep messages readable
                    lines = last_reasoning.strip().splitlines()
                    if len(lines) > 15:
                        display_reasoning = "\n".join(lines[:15])
                        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
                    else:
                        display_reasoning = last_reasoning.strip()
                    # Render style is per-platform: Discord defaults to "-# "
                    # subtext (native small grey metadata text); other
                    # platforms keep the fenced code block.
                    try:
                        from gateway.display_config import resolve_display_setting
                        _reasoning_style = resolve_display_setting(
                            _load_gateway_config(),
                            _platform_config_key(source.platform),
                            "reasoning_style",
                            "code",
                        )
                    except Exception:
                        _reasoning_style = "code"
                    if _reasoning_style == "subtext":
                        _quoted = "\n".join(
                            f"-# {ln}" if ln else "-#" for ln in display_reasoning.splitlines()
                        )
                        response = f"-# 💭 Reasoning\n{_quoted}\n\n{response}"
                    elif _reasoning_style == "blockquote":
                        _quoted = "\n".join(
                            f"> {ln}" if ln else ">" for ln in display_reasoning.splitlines()
                        )
                        response = f"> 💭 **Reasoning:**\n{_quoted}\n\n{response}"
                    else:
                        # Escape ``` inside reasoning so inner fences don't
                        # break the outer code block used to render it.
                        display_reasoning = escape_code_fences_for_display(display_reasoning)
                        response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

            # Runtime-metadata footer — only on the FINAL message of the turn.
            # Off by default (display.runtime_footer.enabled=false).  When
            # streaming already delivered the body, we can't mutate the sent
            # text, so we fire a separate trailing send below.
            _footer_line = ""
            try:
                from gateway.runtime_footer import build_footer_line as _bfl
                _footer_line = _bfl(
                    user_config=_load_gateway_config(),
                    platform_key=_platform_config_key(source.platform),
                    model=agent_result.get("model"),
                    context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                    context_length=agent_result.get("context_length") or None,
                    cwd=os.environ.get("TERMINAL_CWD", ""),
                    turn_seconds=_turn_seconds,
                )
            except Exception as _footer_err:
                logger.debug("runtime_footer build failed: %s", _footer_err)
                _footer_line = ""
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"

            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
                "model": agent_result.get("model", ""),
                "provider": agent_result.get("provider", ""),
            })

            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                # Detach the current batch atomically (see crash-recovery drain
                # above): reassign to a fresh list so a watcher appended by a
                # concurrent session during the yield isn't dropped by clear().
                watchers = process_registry.pending_watchers
                process_registry.pending_watchers = []
                for i, watcher in enumerate(watchers):
                    asyncio.create_task(self._run_process_watcher(watcher))
                    if i % 100 == 99:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Drain watch pattern notifications that arrived during the agent run.
            # Watch events and completions share the same queue; process
            # completions are already handled by the per-process watcher task
            # above, so we only inject watch-type events here.
            #
            # Async-delegation completions ALSO ride this shared queue but are
            # owned by the dedicated _async_delegation_watcher (started at
            # boot), which covers both the idle and post-turn cases with a
            # single consumer — so we leave them on the queue here.
            try:
                from tools.process_registry import process_registry as _pr
                await self._drain_watch_notifications(_pr.completion_queue)
            except Exception as e:
                logger.debug("Watch queue drain error: %s", e)

            # NOTE: Dangerous command approvals are now handled inline by the
            # blocking gateway approval mechanism in tools/approval.py.  The agent
            # thread blocks until the user responds with /approve or /deny, so by
            # the time we reach here the approval has already been resolved.  The
            # old post-loop pop_pending + approval_hint code was removed in favour
            # of the blocking approach that mirrors CLI's synchronous input().

            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            #
            # IMPORTANT: For context-overflow failures (compression exhausted,
            # generic 400 on large sessions) we must NOT persist the user's
            # message — doing so would grow the session further and cause the
            # same failure on the next attempt, an infinite loop. (#1630, #9893)
            #
            # Transient failures (429, timeout, connection error, provider 5xx)
            # are different: the session is not oversized, and silently dropping
            # the user message causes severe context loss on retry — the agent
            # forgets what was just asked.  Persist the user turn so the
            # conversation is preserved. (#7100)
            agent_failed_early = bool(agent_result.get("failed"))
            hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(
                agent_result
            )
            _err_str_for_classify = str(agent_result.get("error", "")).lower()
            # Use specific multi-word phrases (not bare "exceed" or "token")
            # to avoid false positives on transient errors like "rate limit
            # exceeded" or "invalid auth token". Matches run_agent.py's
            # own context-length classifier.
            is_context_overflow_failure = agent_failed_early and (
                bool(agent_result.get("compression_exhausted"))
                or any(p in _err_str_for_classify for p in (
                    "context length", "context size", "context window",
                    "maximum context", "token limit", "too many tokens",
                    "reduce the length", "exceeds the limit",
                    "request entity too large", "prompt is too long",
                    "payload too large", "input is too long",
                ))
                or ("400" in _err_str_for_classify and len(history) > 50)
            )
            if is_context_overflow_failure:
                logger.info(
                    "Skipping transcript persistence for context-overflow "
                    "failure in session %s to prevent session growth loop.",
                    session_entry.session_id,
                )
            elif agent_failed_early:
                logger.info(
                    "Transient agent failure in session %s — persisting user "
                    "message so conversation context is preserved on retry.",
                    session_entry.session_id,
                )
            elif hidden_reasoning_incomplete:
                logger.warning(
                    "Suppressing hidden-reasoning-only incomplete gateway turn "
                    "for session %s: %s",
                    session_entry.session_id,
                    agent_result.get("error", "processing incomplete"),
                )

            # When compression is exhausted, the session is permanently too
            # large to process.  Auto-reset it so the next message starts
            # fresh instead of replaying the same oversized context in an
            # infinite fail loop.  (#9893)
            #
            # A lock-contended defer is the OPPOSITE case: the session is
            # temporarily uncompressible only because a concurrent path holds
            # the compression lock and is actively shrinking it. Never wipe
            # the session for that — retry-next-message semantics apply
            # (#69870 lock-skip consumer; salvaged from #49874).
            if agent_result.get("compression_deferred"):
                logger.info(
                    "Compression deferred for session %s — the compression "
                    "lock is held by a concurrent compressor. Keeping the "
                    "session intact; the next message retries normally.",
                    session_entry.session_id if session_entry else "?",
                )
            elif agent_result.get("compression_exhausted") and session_entry and session_key:
                logger.info(
                    "Auto-resetting session %s after compression exhaustion.",
                    session_entry.session_id,
                )
                new_entry = await self.async_session_store.reset_session(session_key)
                self._evict_cached_agent(session_key)
                # Conversation boundary: one funnel call clears every
                # conversation-scoped per-session dict (#58403 and siblings).
                # See _CONVERSATION_SCOPED_STATE.
                self._clear_conversation_scope(
                    session_key, reason="compression_exhausted_reset"
                )
                if new_entry is not None:
                    # Drop the stale reference to the bloated compressed child and
                    # re-point the Telegram topic binding at the fresh session.
                    # Compression rotated session_entry.session_id to the oversized
                    # compressed child earlier this turn (the agent-result sync
                    # above), and that _sync also rewrote the (chat_id, thread_id)
                    # -> bloated-child binding. reset_session swaps in a clean,
                    # parentless session, but without re-syncing the binding the
                    # next inbound message in this topic gets switch_session'd back
                    # onto the bloated child by the binding-heal walk, reloads the
                    # oversized transcript, and re-triggers compression exhaustion
                    # forever (#35809 — regression of the #9893/#10063 auto-reset).
                    # No-op on non-topic lanes.
                    session_entry = new_entry
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-exhausted-reset",
                    )
                response = (response or "") + (
                    "\n\n🔄 Session auto-reset — the conversation exceeded the "
                    "maximum context size and could not be compressed further. "
                    "Your next message will start a fresh session."
                )

            ts = time.time()  # Unix epoch float — consistent with DB storage

            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if is_context_overflow_failure:
                pass  # Skip all transcript writes — don't grow a broken session
            elif not history:
                tool_defs = agent_result.get("tools", [])
                await self.async_session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": _resolve_gateway_model(),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )

            # The agent already persisted these messages to SQLite via
            # _flush_messages_to_session_db(), so skip the DB write here
            # to prevent the duplicate-write bug (#860 / #42039). This holds
            # for the codex app-server runtime too: although it early-returns
            # and bypasses conversation_loop's per-step flushes, it flushes its
            # own projected assistant/tool messages before returning and
            # reports agent_persisted=True (see agent/codex_runtime.py). Reading
            # the flag (default = self._session_db is not None) keeps the
            # persistence contract explicit and lets any future non-persisting
            # runtime opt into a gateway-side write by returning False.
            agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)

            # Find only the NEW messages from this turn (skip history we loaded).
            # Use the filtered history length (history_offset) that was actually
            # passed to the agent, not len(history) which includes session_meta
            # entries that were stripped before the agent saw them.
            if is_context_overflow_failure:
                pass  # handled above — skip all transcript writes
            elif agent_failed_early or hidden_reasoning_incomplete:
                # Transient failure (429/timeout/5xx): persist only the user
                # message so the next message can load a transcript that
                # reflects what was said.  Skip the assistant error text since
                # it's a gateway-generated hint, not model output. Hidden-
                # reasoning-only incomplete turns follow the same persistence
                # rule so peer-agent channels don't ingest them as completed
                # assistant turns. (#7100, #51628)
                _user_entry = {
                    "role": "user",
                    "content": (
                        persist_user_message
                        if persist_user_message is not None
                        else message_text
                    ),
                    "timestamp": (
                        persist_user_timestamp
                        if persist_user_timestamp is not None
                        else ts
                    ),
                }
                if persist_user_display_kind:
                    _user_entry["display_kind"] = persist_user_display_kind
                if event.message_id:
                    _user_entry["message_id"] = str(event.message_id)
                # Dedupe: skip if this platform message_id is already in the
                # transcript (prevents duplicate user turns on Telegram retries
                # after transient failures). #47237
                _skip_persist = (
                    event.message_id
                    and await self.async_session_store.has_platform_message_id(
                        session_entry.session_id, str(event.message_id)
                    )
                )
                if _skip_persist:
                    logger.info(
                        "Skipping duplicate user turn "
                        "(message_id=%s) in session %s",
                        event.message_id, session_entry.session_id,
                    )
                else:
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
            else:
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []

                # If no new messages found (edge case), fall back to simple user/assistant
                if not new_messages:
                    _user_entry = {
                        "role": "user",
                        "content": (
                            persist_user_message
                            if persist_user_message is not None
                            else message_text
                        ),
                        "timestamp": (
                            persist_user_timestamp
                            if persist_user_timestamp is not None
                            else ts
                        ),
                    }
                    if persist_user_display_kind:
                        _user_entry["display_kind"] = persist_user_display_kind
                    if event.message_id:
                        _user_entry["message_id"] = str(event.message_id)
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
                    if response:
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id,
                            {"role": "assistant", "content": response, "timestamp": ts},
                            skip_db=agent_persisted,
                        )
                else:
                    # Attach the inbound platform message_id to the first user
                    # entry written this turn so platform-level quote-resolution
                    # (e.g. Yuanbao QuoteContextMiddleware's transcript fallback)
                    # can find earlier @bot messages by their original message_id.
                    _user_msg_id_attached = False
                    for msg in new_messages:
                        # Skip system messages (they're rebuilt each run)
                        if msg.get("role") == "system":
                            continue
                        # Add timestamp to each message for debugging
                        entry = {**msg, "timestamp": ts}
                        if (
                            not _user_msg_id_attached
                            and msg.get("role") == "user"
                            and event.message_id
                            and "message_id" not in entry
                        ):
                            entry["message_id"] = str(event.message_id)
                            _user_msg_id_attached = True
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id, entry,
                            skip_db=agent_persisted,
                        )

            # Token counts and model are now persisted by the agent directly.
            # Keep only last_prompt_tokens here for context-window tracking and
            # compression decisions.
            await self.async_session_store.update_session(
                session_entry.session_key,
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
                touch_activity=not bool(getattr(event, "internal", False)),
            )

            # Re-baseline the cached agent's message_count snapshot now that
            # ALL of this turn's transcript writes are done — the agent's
            # flushed user/assistant/tool rows AND the first-turn `session_meta`
            # marker appended above.  The cross-process coherence guard (#45966)
            # snapshots the count at agent-BUILD time (before this turn's own
            # writes) and never refreshes it on reuse, so without this the
            # process's own turn grows message_count and the next turn sees a
            # mismatch and rebuilds the agent — destroying prompt caching.
            #
            # This MUST run after the `session_meta` append: that row also
            # increments message_count, so re-baselining before it (the old
            # position) left the snapshot one short and the guard mis-fired on
            # turn 2 of EVERY fresh gateway conversation, rebuilding the cached
            # agent and busting the prompt cache.  Running here also uses the
            # compaction-updated session_id (the agent_result session_id swap
            # above), matching this function's documented contract.  Refreshing
            # here makes the guard fire only on a DIFFERENT process's writes.
            # Fail-safe inside the helper.
            await self._refresh_agent_cache_message_count(
                session_key, session_entry.session_id
            )

            # Intentional silence is a delivery decision, not a transcript
            # mutation.  The agent's [SILENT]/NO_REPLY assistant turn above is
            # still persisted in session history so later turns keep normal
            # user/assistant alternation; only the outbound chat delivery is
            # suppressed.
            if _intentional_silence:
                logger.info(
                    "Suppressing intentional silence marker for session %s",
                    session_entry.session_id,
                )
                response = ""

            # Auto voice reply: send TTS audio before the text response
            _already_sent = bool(agent_result.get("already_sent"))
            # Skip when streaming TTS already delivered audio for this turn (#60671).
            _stts_adapter = self._adapter_for_source(source)
            _streaming_tts_done = (
                _stts_adapter is not None
                and bool(getattr(_stts_adapter, "_streaming_tts_turn_completed", lambda *_a, **_k: False)(session_key, run_generation))
            )
            if (
                not _streaming_tts_done
                and self._should_send_voice_reply(event, response, agent_messages, already_sent=_already_sent)
            ):
                await self._send_voice_reply(event, response)

            # If streaming already delivered the response, extract and
            # deliver any MEDIA: files before returning None.  Streaming
            # sends raw text chunks that include MEDIA: tags — the normal
            # post-processing in _process_message_background is skipped
            # when already_sent is True, so media files would never be
            # delivered without this.
            #
            # Never skip when the agent failed — the error message is new
            # content the user hasn't seen (streaming only sent earlier
            # partial output before the failure).  Without this guard,
            # users see the agent "stop responding without explanation."
            if agent_result.get("already_sent") and not agent_result.get("failed"):
                if response:
                    _media_adapter = self._adapter_for_source(source)
                    if _media_adapter:
                        await self._deliver_media_from_response(
                            response, event, _media_adapter,
                        )
                # Streaming already delivered the body text, but the footer was
                # intentionally held back (see the `not already_sent` gate above).
                # Send it now as a small trailing message so Telegram/Discord/etc.
                # still surface the runtime metadata on the final reply.
                if _footer_line:
                    try:
                        _foot_adapter = self._adapter_for_source(source)
                        if _foot_adapter:
                            await _foot_adapter.send(
                                source.chat_id,
                                _footer_line,
                                metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
                            )
                    except Exception as _e:
                        logger.debug("trailing footer send failed: %s", _e)
                # This branch returns None so the adapter does not send the
                # body twice. /loop and /goal hooks in _handle_message read
                # the return value, so stash the delivered text on the event
                # or those hooks never run and a /loop tick stays awaiting.
                try:
                    event._streamed_final_response = str(response or "")
                except Exception:
                    pass
                return None

            return response

        except Exception as e:
            # Stop typing indicator on error too, retaining Slack thread/workspace
            # routing so a failed turn cannot leave its status visible.
            try:
                _err_adapter = self._adapter_for_source(source)
                _stop_with_metadata = getattr(
                    type(_err_adapter), "_stop_typing_with_metadata", None
                )
                _stop_typing = getattr(type(_err_adapter), "stop_typing", None)
                if _err_adapter and callable(_stop_with_metadata):
                    await _err_adapter._stop_typing_with_metadata(
                        source.chat_id,
                        self._thread_metadata_for_source(
                            source, self._reply_anchor_for_event(event)
                        ),
                    )
                elif _err_adapter and callable(_stop_typing):
                    await _err_adapter.stop_typing(source.chat_id)
            except Exception:
                pass
            logger.exception("Agent error in session %s", session_key)
            # Crash-resilience for failures that happen before AIAgent enters
            # run_conversation() (for example: provider/httpx client init
            # failures). In that path the agent cannot persist the current
            # inbound turn itself, so append the user message here once. If the
            # agent already reached its early turn-start persistence, the latest
            # transcript user row will match and we skip the duplicate.
            try:
                if 'message_text' in locals() and message_text is not None and session_entry is not None:
                    _already_persisted = False
                    try:
                        _recent_transcript = await self.async_session_store.load_transcript(session_entry.session_id)
                    except Exception:
                        _recent_transcript = []
                    for _msg in reversed(_recent_transcript[-10:]):
                        if _msg.get("role") == "user":
                            _expected_user_content = (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            )
                            _already_persisted = (_msg.get("content") == _expected_user_content)
                            break
                    if not _already_persisted:
                        _user_entry = {
                            "role": "user",
                            "content": (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            ),
                            "timestamp": (
                                persist_user_timestamp
                                if persist_user_timestamp is not None
                                else time.time()
                            ),
                        }
                        if 'persist_user_display_kind' in locals() and persist_user_display_kind:
                            _user_entry["display_kind"] = persist_user_display_kind
                        if getattr(event, "message_id", None):
                            _user_entry["message_id"] = str(event.message_id)
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id,
                            _user_entry,
                        )
            except Exception:
                logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
            # Log full details server-side only; never expose raw exception
            # types or messages to end users (info-leakage risk).
            status_hint = ""
            status_code = getattr(e, "status_code", None)
            _hist_len = len(history) if 'history' in locals() else 0
            if status_code == 401:
                status_hint = " Check your API key or run `claude /login` to refresh OAuth credentials."
            elif status_code == 402:
                status_hint = " Your API balance or quota is exhausted. Check your provider dashboard."
            elif status_code == 429:
                # Check if this is a plan usage limit (resets on a schedule) vs a transient rate limit
                _err_body = getattr(e, "response", None)
                _err_json = {}
                try:
                    if _err_body is not None:
                        _err_json = _err_body.json().get("error", {})
                        if not isinstance(_err_json, dict):
                            _err_json = {}
                except Exception:
                    pass
                if _err_json.get("type") == "usage_limit_reached":
                    _resets_in = _err_json.get("resets_in_seconds")
                    if _resets_in and _resets_in > 0:
                        import math
                        _hours = math.ceil(_resets_in / 3600)
                        status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
                    else:
                        status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
                else:
                    status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif status_code == 529:
                status_hint = " The API is temporarily overloaded. Please try again shortly."
            elif status_code in {400, 500}:
                # 400 with a large session is context overflow.
                # 500 with a large session often means the payload is too large
                # for the API to process — treat it the same way.
                if _hist_len > 50:
                    return (
                        "⚠️ Session too large for the model's context window.\n"
                        "Use /compact to compress the conversation, or "
                        "/reset to start fresh."
                    )
                elif status_code == 400:
                    status_hint = " The request was rejected by the API."
            return (
                f"Sorry, I encountered an unexpected error.{status_hint}\n"
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)

    def _reset_notice_session_info(self, source: SessionSource) -> str:
        """Session-info block for the auto-reset notice, profile-scoped.

        When multiplexing, resolve model/provider/context inside the profile
        serving ``source`` — otherwise the banner advertises the base config's
        model while the session actually runs on the profile's (#59003).
        Mirrors ``_run_agent``'s gating so single-profile gateways never
        enter the scope.

        Call via ``asyncio.to_thread`` from async handlers: under the scope,
        resolution can do blocking work (credential refresh, context-length
        HTTP probes) that must not run on the event loop. The scope is entered
        inside this method, so contextvars behave correctly in the worker
        thread.
        """
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return self._format_session_info()
        return self._format_session_info()

    def _format_session_info(self) -> str:
        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway
        users can immediately see if context detection went wrong (e.g.
        local models falling to the 128K default).
        """
        resolved = _resolve_gateway_model_context()
        model = resolved.model
        provider = resolved.provider
        base_url = resolved.base_url
        context_length = resolved.context_length

        # Format context source hint
        if resolved.context_source == "config":
            ctx_source = "config"
        elif resolved.context_source == "default":
            ctx_source = "default — set model.context_length in config to override"
        else:
            ctx_source = "detected"

        # Format context length for display
        if context_length >= 1_000_000:
            ctx_display = f"{context_length / 1_000_000:.1f}M"
        elif context_length >= 1_000:
            ctx_display = f"{context_length // 1_000}K"
        else:
            ctx_display = str(context_length)

        lines = [
            f"◆ Model: `{model}`",
            f"◆ Provider: {provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]

        # Show endpoint for local/custom setups
        if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
            lines.append(f"◆ Endpoint: {base_url}")

        return "\n".join(lines)




    def _check_slash_access(
        self, source: SessionSource, canonical_cmd: str
    ) -> Optional[str]:
        """Return a denial message if ``source`` cannot run ``canonical_cmd``,
        else None. Used by both the cold and running-agent dispatch paths
        in ``_handle_message`` so admin/user gating can't be bypassed by
        an in-flight agent.

        Backward-compat semantics live in
        :func:`gateway.slash_access.policy_for_source` — when the operator
        hasn't set ``allow_admin_from`` for the scope, the policy returns
        ``enabled=False`` and this method always returns None.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        if not canonical_cmd:
            return None
        policy = _policy_for_source(self.config, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd,
            source.platform.value if source.platform else "?",
            source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: "
                + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "")
                + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this "
                "platform. Ask an admin to add you to allow_admin_from "
                "or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"







    def _sibling_thread_run_keys(self, source: SessionSource, own_key: str) -> list:
        """Find running-agent keys for OTHER participants in the same thread.

        Only applies when the message originates in a thread.  In per-user
        thread mode (``thread_sessions_per_user=True``) each participant gets
        an isolated session key of the form
        ``agent:main:{platform}:{chat_type}:{chat_id}:{thread_id}:{user_id}``,
        so a run started by another user is invisible to the caller's own
        ``/stop``.  This returns the keys of any *actually running* agents
        (not the pending sentinel, not the caller's own key) whose key shares
        the caller's ``{chat_id}:{thread_id}`` prefix.

        Returns an empty list when the source is not in a thread, or when no
        sibling runs exist — callers must still gate on authorization.
        """
        thread_id = getattr(source, "thread_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not thread_id or not chat_id:
            return []
        platform = source.platform.value
        chat_type = getattr(source, "chat_type", None) or ""
        # Prefix that every per-user key in this thread shares, up to and
        # including the thread_id segment.  Matching either the exact
        # shared-thread key or any key with a further (user_id) segment
        # (prefix + ":") avoids cross-matching an unrelated thread whose id
        # merely starts with this one.
        prefix = ":".join(
            ["agent:main", platform, chat_type, str(chat_id), str(thread_id)]
        )
        matches = []
        for key, agent in self._running_agent_items():
            if key == own_key:
                continue
            if agent is _AGENT_PENDING_SENTINEL or not agent:
                continue
            if key == prefix or key.startswith(prefix + ":"):
                matches.append(key)
        return matches




    def _is_stale_restart_redelivery(self, event: MessageEvent) -> bool:
        """Return True if this /restart is a Telegram re-delivery we already handled.

        The previous gateway wrote ``.restart_last_processed.json`` with the
        triggering platform + update_id when it processed the /restart.  If
        we now see a /restart on the same platform with an update_id <= that
        recorded value, it is a redelivery when this process booted from that
        restart. Otherwise the marker must still be recent (< 5 minutes).

        Only applies to Telegram today (the only platform that exposes a
        numeric cross-session update ordering); other platforms return False.
        """
        if event is None or event.source is None:
            return False
        if event.platform_update_id is None:
            return False
        if event.source.platform is None:
            return False
        # Only Telegram populates platform_update_id currently; be explicit
        # so future platforms aren't accidentally gated by this check.
        try:
            platform_value = event.source.platform.value
        except Exception:
            return False
        if platform_value != "telegram":
            return False

        try:
            marker_path = _hermes_home / ".restart_last_processed.json"
            if not marker_path.exists():
                # Belt-and-suspenders for when the dedup marker goes missing
                # (manually cleaned up, or the previous cycle's write failed).
                # Without a marker the update_id comparison below can't run, so
                # a redelivered /restart would sail through and re-restart the
                # gateway — an infinite loop (issue #18528).
                #
                # Suppress ONLY when we can independently confirm we just came
                # out of a restart cycle: this process booted from a
                # chat-originated /restart (_booted_from_restart) AND is still
                # within a short post-boot window. This never swallows a
                # genuine first /restart on a fresh boot (no restart marker on
                # boot → flag stays False). Consume the flag one-shot so a
                # legitimate /restart sent later in the same session is honored.
                if (
                    getattr(self, "_booted_from_restart", False)
                    and time.time() - getattr(self, "_startup_time", 0.0) < 60
                ):
                    self._booted_from_restart = False
                    return True
                return False
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        if data.get("platform") != platform_value:
            return False
        recorded_uid = data.get("update_id")
        if not isinstance(recorded_uid, int):
            return False
        if event.platform_update_id > recorded_uid:
            return False

        # A service-managed restart can legitimately take longer than the
        # marker's normal five-minute trust window while adapters, cron, and
        # in-flight deliveries drain. If this process booted from the recorded
        # chat restart, the first same-or-older update is still that restart's
        # redelivery regardless of elapsed wall time. Consume the boot signal
        # one-shot so a later genuine command is evaluated normally.
        if getattr(self, "_booted_from_restart", False):
            self._booted_from_restart = False
            return True

        # Staleness guard: ignore markers older than 5 minutes.  A legitimately
        # old marker (e.g. crash recovery where notify never fired) should not
        # swallow a fresh /restart from the user.
        requested_at = data.get("requested_at")
        if isinstance(requested_at, (int, float)):
            if time.time() - requested_at > 300:
                return False
        return True









    async def _handle_suggestions_command(self, event: MessageEvent) -> str:
        """Handle /suggestions in the gateway.

        Delegates to the shared handler so CLI and gateway never drift. The
        origin is built from the event source so an accepted suggestion's job
        delivers back to this chat/thread.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command

            return handle_suggestions_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("suggestions command failed: %s", e)
            return f"Suggestions command failed: {e}"

    async def _handle_blueprint_command(self, event: MessageEvent):
        """Handle /blueprint in the gateway.

        Delegates to the shared handler so CLI, TUI, and gateway never drift.
        Returns a BlueprintCommandResult: ``text`` is shown to the user, and if
        ``agent_seed`` is set the dispatch site rewrites ``event.text`` to the
        seed and falls through to the agent (the ``/steer`` pattern) so the
        agent gathers the slot values conversationally. Origin is built from the
        event source so a directly created blueprint job delivers back to this chat.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command

            return handle_blueprint_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("blueprint command failed: %s", e)
            from hermes_cli.blueprint_cmd import BlueprintCommandResult

            return BlueprintCommandResult(f"Cron blueprint command failed: {e}")

    # ────────────────────────────────────────────────────────────────
    # /goal — persistent cross-turn goals (Ralph-style loop)
    # ────────────────────────────────────────────────────────────────
    def _goal_max_turns_from_config(self) -> int:
        """Resolve the configured /goal turn budget for gateway sessions.

        GatewayRunner.config is a GatewayConfig dataclass, not the full
        user config mapping. Top-level config blocks such as ``goals`` are
        therefore only available through hermes_cli.config.load_config().
        """
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    async def _warm_goals_session_db(self, label: str) -> None:
        """Warm the goals SessionDB cache off-loop (best-effort).

        A cold cache runs the state.db init on the loop thread behind the
        bootstrap windows. That freezes the loop for the init duration.
        The executor hop keeps the profile home override alive under
        multiplex, so the warm cache belongs to the caller's profile.
        On failure the caller falls back to the bootstrap windows, so a
        dropped warm-up is a bounded stall, never a crash.
        """
        try:
            from hermes_cli.goals import _get_session_db as _warm_goals_db

            await self._run_in_executor_with_context(_warm_goals_db)
        except Exception as exc:
            logger.warning("%s: session DB warm-up failed: %s", label, exc)

    async def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return a GoalManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)`` or ``(None, None)`` if the
        goals module can't be loaded.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal manager unavailable: %s", exc)
            return None, None
        # Warm the SessionDB cache off-loop. A cold cache freezes the
        # loop for the init duration and drops the first write: the
        # /goal reply claims the goal was set.
        await self._warm_goals_session_db("goal manager")
        try:
            # Session lookups on behalf of an internal event must not advance
            # the user-activity clock that drives idle/daily reset policy
            # (same class as the wake fix in _handle_message_with_agent).
            session_entry = await self.async_session_store.get_or_create_session(
                event.source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("goal manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        max_turns = self._goal_max_turns_from_config()
        return GoalManager(session_id=sid, default_max_turns=max_turns), session_entry

    async def _get_heartbeat_manager_for_event(self, event: "MessageEvent"):
        """Return a HeartbeatManager bound to the session for this event.

        Returns ``(manager, session_entry)`` or ``(None, None)``.
        """
        try:
            from hermes_cli.heartbeat import HeartbeatManager
        except Exception as exc:
            logger.debug("heartbeat manager unavailable: %s", exc)
            return None, None
        # Warm the SessionDB cache off-loop. A cold cache can drop the
        # first /heartbeat write while the reply claims it was set.
        await self._warm_goals_session_db("heartbeat manager")
        try:
            # Same reset-policy contract as _get_goal_manager_for_event:
            # internal events look up the session without touching activity.
            session_entry = await self.async_session_store.get_or_create_session(
                event.source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("heartbeat manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        return HeartbeatManager(session_id=sid), session_entry

    def _register_heartbeat_watch(self, quick_key: str, source: Any, session_id: str) -> None:
        """Track a session with an active heartbeat and start the poller.

        The registry maps ``quick_key`` → ``(source, session_id)`` so the
        poller can rebuild a MessageEvent and enqueue via the adapter FIFO.
        In-memory by design: heartbeat STATE survives restarts in SessionDB,
        but firing resumes when the user touches /heartbeat again in the new
        gateway process (documented; durable schedules belong to cron).
        """
        watch = getattr(self, "_heartbeat_watch", None)
        if watch is None:
            watch = {}
            self._heartbeat_watch = watch
        watch[quick_key] = (source, session_id)
        self._start_heartbeat_poller()

    def _unregister_heartbeat_watch(self, quick_key: str) -> None:
        watch = getattr(self, "_heartbeat_watch", None)
        if watch:
            watch.pop(quick_key, None)

    def _start_heartbeat_poller(self) -> None:
        """Start the single gateway-wide heartbeat poll task (idempotent)."""
        existing = getattr(self, "_heartbeat_poll_task", None)
        if existing is not None and not existing.done():
            return

        from hermes_cli.heartbeat import POLL_SECONDS

        async def _poll_loop():
            while True:
                await asyncio.sleep(POLL_SECONDS)
                watch = getattr(self, "_heartbeat_watch", None)
                if not watch:
                    continue
                # Warm the cache off-loop once per poll. A watch can only
                # be registered through the warmed /heartbeat command, so
                # this covers only the degraded path where that warm-up
                # failed.
                await self._warm_goals_session_db("heartbeat poll")
                for quick_key, (source, session_id) in list(watch.items()):
                    try:
                        # Busy sessions coalesce their tick to the next idle poll.
                        if quick_key in self._running_agents:
                            continue
                        from hermes_cli.heartbeat import HeartbeatManager

                        mgr = HeartbeatManager(session_id=session_id)
                        if not mgr.has_heartbeat():
                            watch.pop(quick_key, None)
                            continue
                        prompt = mgr.due_prompt()
                        if not prompt:
                            continue
                        adapter = self._adapter_for_source(source)
                        if adapter is None:
                            continue
                        hb_event = MessageEvent(
                            text=prompt,
                            message_type=MessageType.TEXT,
                            source=source,
                            message_id=None,
                            channel_prompt=None,
                        )
                        self._enqueue_fifo(quick_key, hb_event, adapter)
                    except Exception as exc:
                        logger.debug("heartbeat poll for %s failed: %s", quick_key, exc)

        try:
            task = asyncio.create_task(_poll_loop())
            self._heartbeat_poll_task = task
            # PERMANENT once started (an infinite while-True loop, no exit
            # condition) — same as a _spawn_supervised watcher. Tag it so
            # _scale_to_zero_has_live_background_work() doesn't treat a
            # gateway with an active heartbeat watch as busy forever.
            task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                _bg.add(task)
                task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start heartbeat poller", exc_info=True)



    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None

        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s",
                getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The gateway message handler returns the agent response to the platform
        adapter, which sends it after this method's caller has returned.  For a
        natural Discord/Telegram reading order, goal status belongs after that
        send.  Platform adapters provide a one-shot post-delivery callback for
        exactly this boundary; when unavailable, fall back to direct awaited
        delivery rather than silently dropping the notice.
        """
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(
                    session_key,
                    _deliver,
                    generation=generation,
                )
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)

        await _deliver()

    async def _post_turn_goal_continuation(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn and, if still active,
        enqueue a continuation prompt for the same session.

        Called from ``_handle_message_with_agent`` at turn boundary, AFTER
        the response has been delivered. Safe when no goal is set.

        We use the adapter's pending-message / FIFO machinery so any real
        user message that arrives simultaneously is handled by the same
        queue and takes priority naturally.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal continuation: goals module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        max_turns = self._goal_max_turns_from_config()

        # Warm the SessionDB cache off-loop. A cold cache runs the
        # state.db init on the loop thread at the turn boundary (the
        # 2026-08-14 crash-loop seam). A slow init can drop the goal
        # read and silently end the goal loop.
        await self._warm_goals_session_db("goal continuation")

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        if not mgr.is_active():
            return

        try:
            from hermes_cli.goals import gather_background_processes as _gather_bg
            _bg_procs = _gather_bg()
        except Exception:
            _bg_procs = None

        # evaluate_after_turn calls judge_goal() which makes a synchronous
        # HTTP request to the auxiliary LLM.  Running it on the event-loop
        # thread would block Discord heartbeats for 10-40 s and cause
        # connection flaps, so we offload it to a thread-pool executor.
        # _run_in_executor_with_context (not bare run_in_executor): the
        # profile secret scope and auxiliary runtime context are contextvars,
        # and a default-executor hop would drop them — aux-client provider
        # resolution would then read credentials unscoped and fail under
        # multiplexing (same pattern as compression in slash_commands.py).
        decision = await self._run_in_executor_with_context(
            lambda: mgr.evaluate_after_turn(
                final_response or "",
                user_initiated=True,
                background_processes=_bg_procs,
            ),
        )
        msg = decision.get("message") or ""

        # Defer the status line until after the adapter has delivered the
        # agent's visible final response. The judge runs after the response is
        # produced but before BasePlatformAdapter sends it, so sending here
        # would show "✓ Goal achieved" before the answer itself. Registering
        # an awaited post-delivery callback preserves delivery reliability
        # without reversing the user-visible ordering.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

        if not decision.get("should_continue"):
            return

        prompt = decision.get("continuation_prompt") or ""
        if not prompt or source is None:
            return

        # Enqueue via the adapter's FIFO so a user message already in
        # flight preempts the continuation naturally.
        try:
            adapter = self._adapter_for_source(source)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                cont_event = MessageEvent(
                    text=prompt,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=None,
                    channel_prompt=None,
                )
                self._enqueue_fifo(_quick_key, cont_event, adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)

    async def _run_post_turn_hooks(
        self,
        *,
        agent_result: Any,
        source: Any,
        is_internal: bool,
        event: Any = None,
    ) -> None:
        """Run goal and loop bookkeeping after an agent turn returns."""
        final_text = self._final_text_for_post_turn_hooks(agent_result, event)

        try:
            session_entry = await self.async_session_store.get_or_create_session(
                source,
                touch_activity=not is_internal,
            )
        except Exception as exc:
            logger.debug("post-turn session resolution failed: %s", exc)
            return

        # Empty interrupted/errored responses must not drive /goal, but an
        # in-flight /loop tick still needs to be released and rescheduled.
        if final_text.strip():
            try:
                await self._post_turn_goal_continuation(
                    session_entry=session_entry,
                    source=source,
                    final_response=final_text,
                )
            except Exception as exc:
                logger.debug("goal continuation hook failed: %s", exc)
        try:
            await self._post_turn_loop_completion(
                session_entry=session_entry,
                source=source,
                final_response=final_text,
            )
        except Exception as exc:
            logger.debug("loop completion hook failed: %s", exc)

    @staticmethod
    def _final_text_for_post_turn_hooks(agent_result, event=None) -> str:
        """Text for /goal and /loop after a gateway turn.

        Streamed turns return None from _handle_message_with_agent
        (already_sent). The delivered reply is stashed on the event so
        those hooks still see it.
        """
        text = ""
        if isinstance(agent_result, dict):
            text = str(agent_result.get("final_response") or "")
        elif isinstance(agent_result, str):
            text = agent_result
        if text.strip():
            return text
        streamed = getattr(event, "_streamed_final_response", None)
        if isinstance(streamed, str) and streamed.strip():
            return streamed
        return text

    async def _post_turn_loop_completion(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Complete a /loop wakeup tick after a gateway turn.

        No-op unless the session has a loop whose tick is in flight
        (``awaiting_response`` — set when the wakeup was injected). Applies
        the LOOP_COMPLETE marker / --until judge / caps and schedules the
        next tick; the idle wakeup watcher fires it when due.
        """
        try:
            from hermes_cli.loops import LoopManager
        except Exception as exc:
            logger.debug("loop completion: loops module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        # Warm the SessionDB cache off-loop. A cold cache at the turn
        # boundary stalls the loop for the init duration and can drop
        # the tick-completion write (the /goal continuation seam, one
        # sibling over).
        await self._warm_goals_session_db("loop completion")

        mgr = LoopManager(session_id=sid)
        state = mgr.state
        if state is None or not state.awaiting_response:
            return

        # The --until judge is a sync aux-LLM call — keep it off the event loop.
        decision = await asyncio.get_running_loop().run_in_executor(
            None, mgr.complete_tick, final_response or ""
        )
        msg = decision.get("message") or ""
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

    async def _loop_wakeup_watcher(self, interval: float = 15.0) -> None:
        """Fire due /loop wakeups for idle gateway sessions.

        The gateway has no per-session scheduler thread, so a coarse ticker
        scans persisted loops (SessionDB ``loop:*`` rows) and injects the
        wakeup prompt into each due session's chat via the same synthetic-
        message path used by watch notifications. Deferrals:

        - session currently running an agent turn → skip (stays due; the
          adapter FIFO would race the live turn otherwise)
        - active non-parked /goal on the session → skip (goal owns the
          idle boundary)
        - no routing metadata on the loop → skip with a one-time warning
          (CLI/TUI loops carry no route and are driven by their own surfaces)
        """
        await asyncio.sleep(5)  # let platforms finish connecting
        warned_no_route: set = set()
        while self._running:
            try:
                from hermes_cli.loops import (
                    LoopManager,
                    goal_blocks_loop_tick,
                    list_active_loops,
                )

                # Warm the cache off-loop once per scan. The scan reads
                # every persisted loop, so a cold cache runs the state.db
                # init on the loop thread before the first read.
                await self._warm_goals_session_db("loop wakeup")

                now = time.time()
                for sid, state in list_active_loops():
                    if state.awaiting_response or now < state.next_due_at:
                        continue
                    route = state.route or {}
                    platform_name = route.get("platform", "")
                    chat_id = route.get("chat_id", "")
                    if not platform_name or not chat_id:
                        # CLI / TUI-owned loop — their own schedulers drive it.
                        continue
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter is None:
                        if sid not in warned_no_route:
                            warned_no_route.add(sid)
                            logger.debug(
                                "loop wakeup: no adapter for platform %r (session %s)",
                                platform_name, sid,
                            )
                        continue

                    # Build the source + session key to check business.
                    evt_stub = {
                        "session_key": "",
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "chat_type": route.get("chat_type", ""),
                        "thread_id": route.get("thread_id", ""),
                        "user_id": route.get("user_id", ""),
                        "user_name": route.get("user_name", ""),
                    }
                    source = self._build_process_event_source(evt_stub)
                    if source is None:
                        continue
                    try:
                        session_key = self._session_key_for_source(source)
                    except Exception:
                        session_key = None
                    if session_key and session_key in self._running_agents:
                        continue  # busy — stays due, next scan retries
                    if goal_blocks_loop_tick(sid):
                        continue

                    mgr = LoopManager(session_id=sid)
                    if not mgr.is_due(now):
                        continue
                    wakeup = mgr.fire_tick()
                    if not wakeup:
                        continue
                    try:
                        synth_event = MessageEvent(
                            text=wakeup,
                            message_type=MessageType.TEXT,
                            source=source,
                            internal=True,
                        )
                        logger.info(
                            "loop wakeup #%s — injecting for %s chat=%s thread=%s",
                            mgr.state.ticks_fired if mgr.state else "?",
                            platform_name, source.chat_id, source.thread_id,
                        )
                        await adapter.handle_message(synth_event)
                        # Slash-command loops dispatch through the command
                        # path and never hit the post-turn completion hook —
                        # complete the tick immediately (caps + scheduling).
                        if wakeup.lstrip().startswith("/"):
                            mgr.complete_tick("")
                    except Exception as exc:
                        logger.warning("loop wakeup injection failed for %s: %s", sid, exc)
                        try:
                            mgr.abandon_tick()
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("loop wakeup watcher error: %s", exc)
            await asyncio.sleep(interval)

    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if raw is None:
            return None
        # Slash command interaction
        if hasattr(raw, "guild_id") and raw.guild_id:
            return int(raw.guild_id)
        # Regular message
        if hasattr(raw, "guild") and raw.guild:
            return raw.guild.id
        return None


    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self._adapter_for_source(event.source)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."

        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."

        voice_channel = await adapter.get_user_voice_channel(
            guild_id, event.source.user_id
        )
        if not voice_channel:
            return "You need to be in a voice channel first."

        # Wire callbacks BEFORE join so voice input arriving immediately
        # after connection is not lost.
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = self._handle_voice_channel_input
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = self._handle_voice_timeout_cleanup
        # Let the adapter's inactivity timer see the live voice-reply mode so it
        # doesn't disconnect a deliberately text-only (/voice off) session.
        if hasattr(adapter, "_voice_mode_getter"):
            adapter._voice_mode_getter = lambda chat_id: self._voice_mode.get(
                self._voice_key(Platform.DISCORD, str(chat_id)), "off"
            )

        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            err_lower = str(e).lower()
            if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
                return (
                    "Voice dependencies are missing (PyNaCl / davey). "
                    f"Install with: `{sys.executable} -m pip install PyNaCl`"
                )
            return f"Failed to join voice channel: {e}"

        if success:
            adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
            if hasattr(adapter, "_voice_sources"):
                adapter._voice_sources[guild_id] = event.source.to_dict()
            self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "all"
            self._save_voice_modes()
            self._set_adapter_auto_tts_enabled(adapter, event.source.chat_id, enabled=True)
            return (
                f"Joined voice channel **{voice_channel.name}**.\n"
                f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
            )
        # Join failed — clear callback
        adapter._voice_input_callback = None
        return "Failed to join voice channel. Check bot permissions (Connect + Speak)."

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self._adapter_for_source(event.source)
        guild_id = self._get_guild_id(event)

        if not guild_id or not hasattr(adapter, "leave_voice_channel"):
            return "Not in a voice channel."

        if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
            return "Not in a voice channel."

        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "off"
        self._save_voice_modes()
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str) -> None:
        """Called by the adapter when a voice channel times out.

        Cleans up runner-side voice_mode state that the adapter cannot reach.
        """
        self._voice_mode[self._voice_key(Platform.DISCORD, chat_id)] = "off"
        self._save_voice_modes()
        adapter = self.adapters.get(Platform.DISCORD)
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    def _is_duplicate_voice_transcript(self, guild_id: int, user_id: int, transcript: str) -> bool:
        """Suppress repeated STT outputs for the same recent utterance.

        Voice capture can occasionally emit the same utterance twice a few
        seconds apart, which creates a second queued agent run and overlapping
        spoken replies. Dedup exact and near-exact repeats per guild/user over a
        short window while allowing genuinely new turns through.
        """
        from difflib import SequenceMatcher

        normalized = re.sub(r"\s+", " ", transcript).strip().lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        if not normalized:
            return False

        now = time.monotonic()
        window_seconds = 12.0
        key = (guild_id, user_id)
        recent_store = getattr(self, "_recent_voice_transcripts", None)
        if not isinstance(recent_store, dict):
            recent_store = {}
            self._recent_voice_transcripts = recent_store
        recent = [
            (ts, txt)
            for ts, txt in recent_store.get(key, [])
            if now - ts <= window_seconds
        ]

        for _, prior in recent:
            if prior == normalized:
                recent_store[key] = recent
                return True
            if len(prior) >= 16 and len(normalized) >= 16:
                if SequenceMatcher(None, prior, normalized).ratio() >= 0.95:
                    recent_store[key] = recent
                    return True

        recent.append((now, normalized))
        recent_store[key] = recent[-5:]
        return False

    @staticmethod
    def _voice_fast_lane_requests_work(transcript: str) -> bool:
        """Return True only for an explicit operational request in voice.

        The voice fast lane is deliberately fail-closed: ordinary conversation
        gets no tool schemas, which avoids model tool loops and makes the
        prompt much smaller. Operators retain a natural escape hatch by plainly
        asking Hermes to inspect, change, test, or otherwise perform work.
        This is a routing guard, not an authorization grant; normal tool
        approvals and platform permissions still apply.
        """
        text = (transcript or "").strip().lower()
        if not text:
            return False
        return bool(
            re.search(
                r"\b(?:analy[sz]e|build|change|check|commit|configure|create|"
                r"debug|deploy|diagnose|edit|fix|implement|inspect|install|"
                r"investigate|look\s+up|merge|patch|pull|push|review|run|"
                r"search|test|update|write)\b",
                text,
            )
        )

    @staticmethod
    def _voice_fast_lane_config() -> dict:
        """Read the optional Discord-only voice fast-lane configuration."""
        try:
            config = _load_gateway_config()
        except Exception:
            return {}
        discord = config.get("discord") if isinstance(config, dict) else None
        if not isinstance(discord, dict):
            return {}
        policy = discord.get("voice_fast_lane")
        return dict(policy) if isinstance(policy, dict) else {}

    def _voice_fast_lane_matches(
        self, adapter, guild_id: int, user_id: int
    ) -> tuple[bool, str]:
        """Match an inbound speaker to the configured Discord voice fast lane.

        Matching requires the exact configured voice channel and an allowlisted
        speaker. Missing/malformed config or incomplete Discord state disables
        the lane rather than broadening it to another guild/channel.
        """
        policy = self._voice_fast_lane_config()
        if policy.get("enabled") is not True:
            return False, ""
        channel_id = str(policy.get("channel_id") or "").strip()
        user_ids = policy.get("user_ids")
        if not channel_id or not isinstance(user_ids, list):
            return False, ""
        if str(user_id) not in {str(item) for item in user_ids}:
            return False, ""
        clients = getattr(adapter, "_voice_clients", None)
        client = clients.get(guild_id) if isinstance(clients, dict) else None
        actual_channel_id = str(getattr(getattr(client, "channel", None), "id", "") or "")
        if actual_channel_id != channel_id:
            return False, ""
        return True, channel_id

    @staticmethod
    def _is_voice_fast_lane_event(event: MessageEvent) -> bool:
        return bool(
            getattr(event, "message_type", None) == MessageType.VOICE
            and getattr(getattr(event, "source", None), "_voice_fast_lane", False)
        )

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str
    ):
        """Handle transcribed voice from a user in a voice channel.

        Creates a synthetic MessageEvent and processes it through the
        adapter's full message pipeline (session, typing, agent, TTS reply).
        """
        adapter = self.adapters.get(Platform.DISCORD)
        if not adapter:
            return

        text_ch_id = adapter._voice_text_channels.get(guild_id)
        if not text_ch_id:
            return

        # Build source — reuse the linked text channel's metadata when available
        # so voice input shares the same session as the bound text conversation.
        source_data = getattr(adapter, "_voice_sources", {}).get(guild_id)
        if source_data:
            source = SessionSource.from_dict(source_data)
            source.user_id = str(user_id)
            source.user_name = str(user_id)
        else:
            source = SessionSource(
                platform=Platform.DISCORD,
                chat_id=str(text_ch_id),
                user_id=str(user_id),
                user_name=str(user_id),
                chat_type="channel",
            )

        # Check authorization before processing voice input
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return

        if self._is_duplicate_voice_transcript(guild_id, user_id, transcript):
            logger.info(
                "Suppressing duplicate voice transcript for guild=%s user=%s: %s",
                guild_id,
                user_id,
                transcript[:100],
            )
            return

        # Show transcript in text channel (after auth, with mention sanitization)
        try:
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        except Exception:
            pass

        # Build a synthetic MessageEvent and feed through the normal pipeline
        # Use SimpleNamespace as raw_message so _get_guild_id() can extract
        # guild_id and _send_voice_reply() plays audio in the voice channel.
        from types import SimpleNamespace
        # Resolve the bound text channel's channel_prompt so voice input gets
        # the same per-channel context as typed messages (#50149).
        channel_prompt: Optional[str] = None
        resolver = getattr(adapter, "_resolve_channel_prompt", None)
        if callable(resolver):
            try:
                resolved = resolver(str(text_ch_id))
                channel_prompt = resolved if isinstance(resolved, str) else None
            except Exception:
                channel_prompt = None
        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
            channel_prompt=channel_prompt,
        )

        fast_lane, voice_channel_id = self._voice_fast_lane_matches(
            adapter, guild_id, user_id
        )
        if fast_lane:
            # Keep delivery in the bound text channel while assigning this
            # spoken conversation its own small session. These are private
            # transport hints and intentionally do not serialize with Source.
            source._voice_fast_lane = True
            source._session_key_lane = f"discord-voice:{voice_channel_id}"
            event.metadata["voice_fast_lane"] = True
            logger.info(
                "Voice fast lane enabled for guild=%s channel=%s user=%s",
                guild_id,
                voice_channel_id,
                user_id,
            )

        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
        already_sent: bool = False,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          UNLESS streaming already consumed the response (already_sent=True),
          in which case the base adapter won't have text for auto-TTS so the
          runner must handle it.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_key = self._voice_key(event.source.platform, chat_id)
        voice_mode = self._voice_mode.get(voice_key)
        is_voice_input = (event.message_type == MessageType.VOICE)

        adapter = self.adapters.get(event.source.platform)
        adapter_auto_tts = False
        if adapter and hasattr(adapter, "_should_auto_tts_for_chat"):
            try:
                adapter_auto_tts = bool(adapter._should_auto_tts_for_chat(chat_id))
            except Exception:
                adapter_auto_tts = False

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
            # ``voice.auto_tts`` is synced into the adapter on gateway startup.
            # It is the fallback only when the chat has no explicit mode;
            # otherwise the chat-level all/voice_only/off choice takes precedence.
            or (voice_mode is None and adapter_auto_tts)
        )
        if not should:
            logger.debug(
                "Auto voice reply skipped: mode=%s adapter_auto_tts=%s chat=%s platform=%s",
                voice_mode, adapter_auto_tts, chat_id, event.source.platform.value,
            )
            return False

        # Dedup: agent already called TTS tool in THIS turn only
        last_user_idx = None
        for i, msg in enumerate(reversed(agent_messages)):
            if msg.get("role") == "user":
                last_user_idx = len(agent_messages) - 1 - i; break
        turn_messages = agent_messages[last_user_idx:] if last_user_idx is not None else agent_messages
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                (tc.get("function") or {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in turn_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        # When streaming already delivered the text (already_sent=True),
        # the base adapter will receive None and can't run auto-TTS,
        # so the runner must take over.
        if is_voice_input and not already_sent:
            return False

        return True

    def _should_echo_stt_transcripts(self) -> bool:
        """Return whether inbound voice/STT transcripts should be echoed to chat."""
        return bool(getattr(self.config, "stt_echo_transcripts", True))

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        audio_path = None
        actual_paths: List[str] = []
        try:
            from tools.tts_tool import text_to_speech_tool
            from tools.tts_text_normalize import _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text)
            if not tts_text:
                return

            # Platform-aware output path: platforms whose native voice
            # bubbles require Ogg/Opus (OPUS_VOICE_PLATFORMS — Telegram,
            # Matrix, Feishu, WhatsApp, Signal) get an explicit .ogg path;
            # the TTS tool's central container repair guarantees real
            # Ogg/Opus bytes for every provider. Others keep MP3.
            audio_path = build_auto_tts_output_path(event.source.platform)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            try:
                result = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Auto voice reply TTS returned invalid JSON: %s", result_json[:200] if result_json else result_json)
                return

            # Final delivery may be one combined file or multiple separately
            # valid files when combination is unavailable or would exceed a
            # platform limit. Preserve legacy single-file results.
            actual_paths = result.get("file_paths") or [
                result.get("file_path", audio_path)
            ]
            actual_paths = [
                str(path) for path in actual_paths
                if path and os.path.isfile(path)
            ]
            if not result.get("success") or not actual_paths:
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self._adapter_for_source(event.source)

            # If connected to a voice channel, play there instead of sending a file
            guild_id = self._get_guild_id(event)
            play_in_voice_channel = getattr(adapter, "play_in_voice_channel", None)
            is_in_voice_channel = getattr(adapter, "is_in_voice_channel", None)
            send_voice = getattr(adapter, "send_voice", None)
            in_voice_channel = bool(
                guild_id
                and callable(play_in_voice_channel)
                and callable(is_in_voice_channel)
                and is_in_voice_channel(guild_id)
            )
            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
            if not in_voice_channel and callable(send_voice):
                # Mark the auto voice reply as notify-worthy.  Mirrors the
                # final-text path in gateway/platforms/base.py which sets
                # ``notify=True`` so platform adapters that gate push
                # notifications (Telegram "important" mode) deliver the
                # final voice reply as a normal notification instead of a
                # silent message.  Clone first so we don't mutate metadata
                # shared with concurrent typing-indicator state.
                if thread_meta is not None:
                    thread_meta = dict(thread_meta)
                    thread_meta["notify"] = True
                else:
                    thread_meta = {"notify": True}
            for actual_path in actual_paths:
                if in_voice_channel:
                    play_voice = cast(Callable[..., Awaitable[Any]], play_in_voice_channel)
                    await play_voice(guild_id, actual_path)
                elif callable(send_voice):
                    send_voice_call = cast(Callable[..., Awaitable[Any]], send_voice)
                    send_kwargs: Dict[str, Any] = {
                        "chat_id": event.source.chat_id,
                        "audio_path": actual_path,
                        "reply_to": reply_anchor,
                        "metadata": thread_meta,
                    }
                    await send_voice_call(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in ({audio_path, *actual_paths} - {None}):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
        thread_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract explicit MEDIA: tags from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.

        Unlike the non-streaming path in ``gateway/platforms/base.py`` (which
        also auto-detects bare local paths via ``extract_local_files``), this
        post-stream rescan is EXPLICIT-ONLY. The visible reply has already
        been streamed verbatim, so a bare path string here was either (a)
        already shown to the user as text, or (b) stale tool/inspected
        content that was never part of the intended visible reply. Promoting
        such paths into uploads after the fact sent files the model never
        asked to deliver (#20834). Only ``MEDIA:`` directives — the explicit
        attachment contract — trigger post-stream uploads.
        """
        from pathlib import Path
        from urllib.parse import quote as _quote

        try:
            # Capture [[as_document]] before extract_media strips it, so the
            # dispatch partition below can route image-extension files
            # through send_document (preserving bytes) instead of
            # send_multiple_images (Telegram sendPhoto recompresses to ~1280px).
            force_document_attachments = "[[as_document]]" in response

            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            # Do NOT deduplicate explicit MEDIA tags against prior turns here
            # (#73771). This rescan is already EXPLICIT-ONLY (see docstring):
            # a MEDIA: directive in the final streamed reply is the model
            # deliberately attaching a file — including a user-requested
            # resend. Stale auto-appended tags are deduped upstream in
            # _collect_auto_append_media_tags with history_media_paths.
            # Mirrors the same filter removal on the non-streaming path in
            # gateway/platforms/base.py.
            # Strip image URLs from the cleaned text for parity with the
            # non-streaming chain, but do NOT run extract_local_files here:
            # post-stream delivery is explicit-only (#20834). Bare local paths
            # in an already-streamed reply are text the user has seen (or
            # stale inspected content), not an attachment request.
            adapter.extract_images(cleaned)

            _thread_meta = (
                dict(thread_metadata)
                if thread_metadata is not None
                else self._thread_metadata_for_source(
                    event.source,
                    self._reply_anchor_for_event(event),
                )
            )

            _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
            _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

            # Partition out images so they can be sent as a single batch
            # (e.g. Signal's multi-attachment RPC). When [[as_document]] was
            # set, image-extension files skip the photo path and route to
            # send_document below — preserving original bytes.
            image_paths: list = []
            non_image_media: list = []
            for media_path, is_voice in media_files:
                ext = Path(media_path).suffix.lower()
                if (ext in _IMAGE_EXTS
                        and not is_voice
                        and not force_document_attachments):
                    image_paths.append(media_path)
                else:
                    non_image_media.append((media_path, is_voice))

            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(
                        chat_id=event.source.chat_id,
                        images=images,
                        metadata=_thread_meta,
                    )
                except Exception as e:
                    logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)

            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=media_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=media_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

        except Exception as e:
            logger.warning("Post-stream media extraction failed: %s", e)

    async def _deliver_queued_first_response(
        self,
        response: str,
        source: SessionSource,
        adapter,
        metadata: Optional[Dict[str, Any]] = None,
        event_message_id: Optional[str] = None,
        text_already_delivered: bool = False,
        deliver_media: bool = True,
        stream_consumer=None,
    ) -> None:
        """Deliver a queued response using the normal text+attachment split."""
        if not text_already_delivered:
            text_content = _strip_response_attachments_for_direct_send(response, adapter)
            if text_content:
                # Reconcile-by-edit first (live finding, 2026-08-16 canary):
                # when the stream consumer delivered/sealed a message but its
                # recorded payload didn't confirm the final (post-stream
                # mutation), plain-sending here creates the duplicate — the
                # sealed message already carries most of the answer. A sealed
                # native stream is a regular message; chat.update on it is
                # live-verified working. Fall back to plain send only when
                # there is no editable message or the edit fails.
                _reconciled = False
                _sc_msg_id = getattr(stream_consumer, "message_id", None)
                if (
                    _sc_msg_id
                    and _sc_msg_id != "__no_edit__"
                    and not getattr(stream_consumer, "_turn_split_delivery", False)
                ):
                    try:
                        _edit_res = await adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=text_content,
                            finalize=True,
                        )
                        if getattr(_edit_res, "success", False):
                            _reconciled = True
                            logger.info(
                                "Queued-lane final reconciled by editing message %s in place (no duplicate send).",
                                _sc_msg_id,
                            )
                    except Exception as _qe:
                        logger.debug(
                            "Queued-lane reconcile edit failed (%s); falling back to send.",
                            _qe,
                        )
                if not _reconciled:
                    await adapter.send(
                        source.chat_id,
                        text_content,
                        metadata=metadata,
                    )

        # Failed turns still deliver their (normalized failure) text above,
        # but must not upload attachments as if the turn succeeded — mirrors
        # the ``not agent_result.get("failed")`` guard on the completed-turn
        # delivery path.
        if not deliver_media:
            return

        synthetic_event = MessageEvent(
            text="",
            source=source,
            message_id=event_message_id,
        )
        await self._deliver_media_from_response(
            response,
            synthetic_event,
            adapter,
            thread_metadata=metadata,
        )

    async def _run_background_task(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Profile-scoping wrapper around the background agent task.

        When multiplexing is active, resolve the inbound source's profile and
        run the whole task inside ``_profile_runtime_scope`` so credentials
        resolve from that profile's secret scope. Mirrors the pattern in
        ``_run_agent``.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

    def _resolve_enabled_toolsets_for_source(
        self,
        user_config: dict,
        source: "SessionSource",
        platform_key: str,
    ) -> list:
        """Resolve enabled toolsets for an agent run, honoring per-source overrides.

        Asks the receiving adapter for a ``toolsets_for_source()`` override
        (e.g. per-route webhook toolsets). When present, the override list is
        validated through the SAME ``_get_platform_tools`` path as normal
        platform config — by substituting it as the platform's toolset list —
        so unknown names and platform-restricted toolsets are dropped rather
        than trusted. When absent, falls back to standard
        ``platform_toolsets.<platform>`` resolution.
        """
        from hermes_cli.tools_config import _get_platform_tools

        override = None
        try:
            adapter = self._adapter_for_source(source)
            if adapter is not None:
                override = adapter.toolsets_for_source(source)
        except Exception:
            override = None

        if override and isinstance(override, list):
            cfg = dict(user_config)
            pts = dict(cfg.get("platform_toolsets") or {})
            pts[platform_key] = [str(t) for t in override]
            cfg["platform_toolsets"] = pts
            return sorted(_get_platform_tools(cfg, platform_key))

        return sorted(_get_platform_tools(user_config, platform_key))

    async def _run_background_task_inner(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from run_agent import AIAgent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            enabled_toolsets = self._resolve_enabled_toolsets_for_source(
                user_config, source, platform_key
            )
            agent_cfg = user_config.get("agent") or {}
            from agent.skill_utils import parse_config_string_list

            disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None

            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(
                source=source, model=model
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._resolve_session_service_tier(source=source)
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    **_checkpoint_agent_kwargs(user_config),
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    # Reload from disk — do not reuse the startup snapshot (#60955).
                    fallback_model=self._refresh_fallback_model(),
                )
                try:
                    return agent.run_conversation(
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Background tasks start a fresh conversation (no prior history),
            # so history_offset=0: every message in the run belongs to this
            # turn. Mirrors the repair on the main turn path.
            if response:
                response = repair_explicit_computer_use_media_paths(
                    response,
                    result.get("messages", []),
                )

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )
                    except Exception:
                        pass

                # Send media files, routing each by type so a TTS clip
                # arrives as a voice bubble / a clip as a video rather than
                # a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass







    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        adapter = self._adapter_for_source(source)
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            if isinstance(me, dict):
                return me.get(name)
            return None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id:
            return

        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return

        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return

        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Return a Bot API-safe forum topic name from a generated session title."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        # Telegram forum topic names are short (currently 1-128 chars). Keep
        # extra room for multi-byte titles and avoid trailing ellipsis churn.
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    def _is_discord_auto_thread_lane(self, source: SessionSource) -> bool:
        """Return True only for Discord threads Hermes just auto-created."""
        return (
            source.platform == Platform.DISCORD
            and source.chat_type == "thread"
            and bool(getattr(source, "auto_thread_created", False))
            and bool(source.thread_id)
            and bool(getattr(source, "auto_thread_initial_name", None))
        )

    def _is_relay_discord_channel_lane(self, source: SessionSource) -> bool:
        """Shape-only check: a relay-delivered Discord CHANNEL event whose
        reply the connector MAY auto-thread (title-turn registration gate).

        Deliberately does NOT consult the send-result cache: at registration
        time (before delivery) the feedback can't exist yet. The rename lane
        polls the cache at fire time instead."""
        return (
            source.platform == Platform.DISCORD
            and bool(source.chat_id)
            and not source.thread_id
            and source.chat_type in ("group", "channel")
            and getattr(source, "delivered_via_upstream_relay", False) is True
        )

    def _relay_auto_thread_info(
        self, source: SessionSource
    ) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) when the RELAY connector auto-threaded our
        reply to this source's chat — the title-turn sibling of
        _is_discord_auto_thread_lane.

        The marker-based check above only lights up for events ARRIVING IN an
        auto-created thread (turn 2+). The auto-title fires on the FIRST
        exchange, whose source is the PARENT channel event — the thread did
        not exist at ingest, so no markers can be present and the native lane
        check never matches on the relay title turn (staging repro
        2026-07-29: initial titles fine, semantic renames never happened).

        Preferred path: the connector stamps ``prospective_thread_id`` on the
        inbound (the anchor message id, which IS the id of the thread it will
        auto-create). It's deterministic and per-message, so it identifies the
        EXACT thread even when several auto-threads spawn from one channel —
        unlike the send-result cache below, which held a single slot per parent
        chat and so only the FIRST thread in a channel ever renamed (staging
        repro 2026-08-02: thread A renamed, sibling thread B stuck at raw
        text). The connector's own created-name guard (prefer_connector_created)
        enforces no-clobber, so no initial name is needed here.

        Fallback: the connector reports where the reply actually landed on the
        send result (contract §SendResult thread_id/auto_thread_name); the
        relay adapter caches it per chat and this reads it back — kept for
        older connectors that don't stamp prospective_thread_id.
        """
        if source.platform != Platform.DISCORD or not source.chat_id:
            return None
        if not getattr(source, "delivered_via_upstream_relay", False):
            return None
        prospective = getattr(source, "prospective_thread_id", None)
        if prospective:
            # Deterministic per-thread identity; the empty initial-name marker
            # signals the caller to rely on the connector-side no-clobber guard.
            return (str(prospective), "")
        adapter = self._adapter_for_source(source)
        info_fn = getattr(adapter, "auto_thread_info_for_chat", None)
        if not callable(info_fn):
            return None
        try:
            return _as_thread_info(info_fn(str(source.chat_id)))
        except Exception:
            return None

    async def _await_relay_auto_thread_info(
        self, source: SessionSource
    ) -> Optional[Tuple[str, str]]:
        """``_relay_auto_thread_info``, waited out until this turn delivers.

        The legacy send-result path can only answer once the reply is sent, and
        the caller asks at title time — one turn early. The adapter answers on
        the send either way, so the timeout is only a backstop for a turn that
        never sends at all; the turn's own inactivity limit is exactly how long
        that turn could still be alive.
        """
        # The connector-stamped prospective id is known at ingest, so most
        # sessions answer here and never wait at all.
        known = self._relay_auto_thread_info(source)
        if known is not None:
            return known
        adapter = self._adapter_for_source(source)
        wait_fn = getattr(adapter, "wait_for_auto_thread_info", None)
        if not callable(wait_fn) or not source.chat_id:
            return None
        # 0 means the operator disabled the turn limit; the backstop still needs one.
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800) or 1800
        try:
            return _as_thread_info(await wait_fn(str(source.chat_id), timeout))
        except Exception:
            return None

    def _sanitize_discord_thread_title(self, title: str) -> str:
        """Return a Discord-safe semantic thread title from a session title.

        Discord thread names are capped at 100 characters measured in UTF-16
        code units (emoji count double), so truncate with the UTF-16 helpers
        rather than Python code-point slices.
        """
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        if utf16_len(cleaned) > 80:
            cleaned = _prefix_within_utf16_limit(cleaned, 77).rstrip() + "..."
        return cleaned

    async def _rename_discord_auto_thread_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
        relay_info: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Best-effort semantic rename of a newly auto-created Discord thread.

        ``relay_info`` is the (thread_id, initial_name) pair from the relay
        connector's send-result feedback — supplied on the title turn, where
        the source is the parent-channel event and carries no auto-thread
        markers (see _relay_auto_thread_info). When absent, the native
        marker-based lane supplies thread identity from the source itself.
        """
        if relay_info is None and not await asyncio.to_thread(
            self._is_discord_auto_thread_lane, source
        ):
            # Relay title turn with no feedback captured at schedule time: the
            # title comes off the user's opening message, so it beats the
            # delivery that produces the connector's send-result feedback
            # (thread_id + initial name) by the whole length of the turn. Wait
            # on the adapter for that send rather than guessing how long the
            # turn will take.
            if not self._is_relay_discord_channel_lane(source):
                return
            relay_info = await self._await_relay_auto_thread_info(source)
            if relay_info is None:
                # True miss: the connector did not auto-thread this reply
                # (policy off, DM, already-threaded, or send failed).
                return
        adapter = self._adapter_for_source(source) if getattr(self, "adapters", None) else None
        if adapter is None:
            return
        rename_thread = getattr(adapter, "rename_thread", None)
        if rename_thread is None:
            return
        target_thread_id = relay_info[0] if relay_info else str(source.thread_id)
        # Relay lane (relay_info present): ask the CONNECTOR to enforce the
        # no-clobber guard from its own created-name memory — the gateway
        # can't reliably reproduce the thread's initial name byte-for-byte
        # (normalization drift silently declined every rename before this).
        # Native-marker lane keeps the legacy string guard.
        use_connector_guard = relay_info is not None
        guard_name = (
            None
            if use_connector_guard
            else getattr(source, "auto_thread_initial_name", None)
        )
        thread_name = self._sanitize_discord_thread_title(title)
        # Relay lane only: the connector's egress guard resolves the owning
        # tenant from the outbound metadata's scope_id (guild) / user_id
        # (author). Those discriminator caches are keyed by the PARENT channel
        # chat_id (learned at inbound), NOT the thread id. rename_thread
        # defaults chat_id to the thread id when no parent is given, so the
        # scope/author lookup misses and the connector declines the op
        # ("target not routed to an onboarded tenant" — the live failure on
        # staging 2026-08-01). Pass the parent channel id (the relay source's
        # chat_id IS the parent channel; the thread came from send-result
        # feedback) so the discriminators resolve. Native lane needs nothing:
        # its source IS the thread and it renames via the direct Discord API,
        # not the relay egress guard.
        parent_chat_id = (
            str(source.chat_id) if use_connector_guard and source.chat_id else None
        )
        logger.info(
            "discord auto-thread rename: thread=%s lane=%s new_title=%r",
            target_thread_id,
            "relay" if use_connector_guard else "native",
            thread_name,
        )
        try:
            renamed = await rename_thread(
                target_thread_id,
                thread_name,
                prefer_connector_created=use_connector_guard,
                only_if_current_name=guard_name,
                parent_chat_id=parent_chat_id,
            )
            logger.info(
                "discord auto-thread rename result: thread=%s applied=%s",
                target_thread_id,
                bool(renamed),
            )
        except Exception:
            logger.debug("Failed to rename Discord auto-thread for generated session title", exc_info=True)

    def _schedule_discord_semantic_thread_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule Discord auto-thread rename from the auto-title background thread."""
        relay_info = None
        if not title:
            return
        if not self._is_discord_auto_thread_lane(source):
            # Relay title turn: the source is the PARENT channel event (the
            # thread didn't exist at ingest, so no auto-thread markers). The
            # connector's send-result feedback tells us where the reply
            # landed — but the auto-title thread races the delivery that
            # produces it, so a cache miss HERE is not a verdict. Schedule
            # whenever the SHAPE matches; the async rename lane polls the
            # cache (with a bounded wait) and no-ops on a true miss.
            relay_info = self._relay_auto_thread_info(source)
            if relay_info is None and not self._is_relay_discord_channel_lane(
                source
            ):
                return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_discord_auto_thread_for_session_title(
                copied_source, session_id, title, relay_info=relay_info
            ),
            loop,
            logger=logger,
            log_message="Discord semantic thread rename failed to schedule",
        )
        if future is None:
            return

        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Discord semantic thread rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    async def _rename_telegram_topic_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return

        # Operator can fully disable per-topic auto-rename via
        # extra.disable_topic_auto_rename. Useful when topics are managed
        # by the user (ad-hoc Threaded Mode) and auto-rename would
        # overwrite their chosen names every time the auto-title fires.
        if self._telegram_topic_auto_rename_disabled(source):
            return

        # Skip rename when the topic is operator-declared via
        # extra.dm_topics. Those topics have fixed names chosen by the
        # operator (plus optional skill binding); auto-renaming would
        # silently mutate operator config.
        #
        # Check the class, not the instance — getattr() on MagicMock
        # auto-creates attributes, so `hasattr(adapter, "_get_dm_topic_info")`
        # would return True for every test double.
        adapter = self._adapter_for_source(source)
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only treat dict-shaped returns as operator-declared; a
                # bare MagicMock or other sentinel shouldn't count.
                if isinstance(operator_topic, dict):
                    return

        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return

        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    name=topic_name,
                )
                return

            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) if bot is not None else None
            if edit_forum_topic is None:
                edit_forum_topic = getattr(bot, "editForumTopic", None) if bot is not None else None
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id),
                    message_thread_id=int(source.thread_id),
                    name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id,
                    message_thread_id=source.thread_id,
                    name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """Return True when operator disabled per-topic auto-rename for this Telegram chat.

        Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
        Default is False (auto-rename enabled, preserves prior behaviour).
        """
        platform_cfg = (
            self.config.platforms.get(source.platform)
            if getattr(self, "config", None) and getattr(self.config, "platforms", None)
            else None
        )
        if platform_cfg is None:
            return False
        extra = getattr(platform_cfg, "extra", None) or {}
        value = extra.get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _schedule_telegram_topic_title_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_telegram_topic_for_session_title(copied_source, session_id, title),
            loop,
            logger=logger,
            log_message="Telegram topic title rename failed to schedule",
        )
        if future is None:
            return
        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Telegram topic title rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot.

        If a user sends /topic repeatedly while Threads Settings are still
        off, we shouldn't keep re-uploading the screenshot every time.
        """
        if not hasattr(self, "_telegram_capability_hint_ts"):
            self._telegram_capability_hint_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_capability_hint_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S:
            return False
        self._telegram_capability_hint_ts[chat_id] = now
        return True

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        # No-op if never enabled.
        try:
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id,
                user_id=str(source.user_id or ""),
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-chat debounce state so the user doesn't see a stale
        # cooldown on the next activation.
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(chat_id, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )


    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []

        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                session_id = str(session.get("id") or "")
                title = str(session.get("title") or "Untitled session")
                preview = str(session.get("preview") or "").strip()
                line = f"- {title} — `{session_id}`"
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        session_id = await self._session_db.resolve_session_id(raw_session_id.strip())
        if not session_id:
            return f"Session not found: {raw_session_id.strip()}"

        session = await self._session_db.get_session(session_id)
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."

        linked = await self._session_db.is_telegram_session_linked_to_topic(session_id=session_id)
        current_binding = await self._session_db.get_telegram_topic_binding(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
        )
        if linked:
            if not current_binding or current_binding.get("session_id") != session_id:
                return "That session is already linked to another Telegram topic."

        session_key = self._session_key_for_source(source)
        try:
            await self._session_db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=session_key,
                session_id=session_id,
                managed_mode="restored",
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return "That session is already linked to another Telegram topic."
            raise

        title = await self._session_db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(await self._session_db.get_messages(session_id)):
                if message.get("role") != "assistant":
                    continue
                projected = project_compaction_message_for_display(message)
                if projected is not None and projected.get("content"):
                    last_assistant = str(projected.get("content"))
                    break
        except Exception:
            last_assistant = None

        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response







    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Actually disconnect, reconnect, and notify MCP tool changes.

        Split out from ``_handle_reload_mcp_command`` so the confirmation
        wrapper can invoke the same path whether the user confirmed via
        button, text reply, or has the confirm gate disabled.
        """
        loop = asyncio.get_running_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = [t("gateway.reload_mcp.header")]
            if reconnected:
                lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
            if added:
                lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
            if removed:
                lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            # Refresh cached agents so existing sessions see new MCP tools on
            # their next turn — without this, the user has to `/new` (which
            # discards conversation history) to pick up tools from a server
            # that was just added or reconnected. The user has already
            # consented to the prompt-cache invalidation via the slash-confirm
            # gate in _handle_reload_mcp_command before we reach this point.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools
                _cache = getattr(self, "_agent_cache", None)
                _cache_lock = getattr(self, "_agent_cache_lock", None)
                if _cache_lock is not None and _cache:
                    with _cache_lock:
                        for _sess_key, _entry in list(_cache.items()):
                            try:
                                _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                            except Exception:
                                continue
                            if _agent is None:
                                continue
                            # Preserve each cached agent's build-time toolset
                            # selection EXACTLY: a gateway session built with a
                            # restricted enabled_toolsets (e.g. ["safe"]) must
                            # NOT silently gain tools after a reload. This is the
                            # opposite of the interactive CLI/TUI /reload-mcp,
                            # which is a single user re-applying their own config
                            # edit; gateway agents are per-session and may be
                            # deliberately locked down. (Contract is asserted by
                            # test_reload_mcp_preserves_per_agent_toolset_overrides.)
                            refresh_agent_mcp_tools(_agent, quiet_mode=True)
            except Exception as _exc:
                logger.debug(
                    "Failed to update cached agent tools after MCP reload: %s",
                    _exc,
                )

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)



    # ------------------------------------------------------------------
    # Slash-command confirmation primitive (generic)
    # ------------------------------------------------------------------
    # Used by slash commands that have a non-destructive but expensive
    # side effect worth an explicit user confirmation (currently only
    # /reload-mcp, which invalidates the prompt cache).  Two delivery
    # paths:
    #   1. Button UI — adapters that override ``send_slash_confirm``
    #      (Telegram, Discord, Slack, Matrix, Feishu) render three
    #      inline buttons.  The adapter routes the button click back via
    #      ``tools.slash_confirm.resolve(session_key, confirm_id, choice)``.
    #   2. Text fallback — adapters that don't override the hook get a
    #      plain text prompt.  Users reply with /approve, /always, or
    #      /cancel; the early intercept in ``_handle_message`` matches
    #      those replies against ``tools.slash_confirm.get_pending()``.

    async def _maybe_confirm_destructive_slash(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        detail: str,
        execute,
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute`` is an async callable ``execute() -> str | EphemeralReply``
        that performs the destructive action.  If the
        ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
        runs immediately (returning its result).  Otherwise this routes
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``
        """
        # Gate check.
        confirm_required = True
        try:
            cfg = self._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            if choice == "cancel":
                return f"🟡 /{command} cancelled. Conversation unchanged."
            persisted = False
            if choice == "always":
                try:
                    from cli import save_config_value
                    # save_config_value swallows its own errors and reports the
                    # outcome in the return value, so the try block alone says
                    # nothing about whether the write landed.
                    persisted = bool(
                        save_config_value("approvals.destructive_slash_confirm", False)
                    )
                    if persisted:
                        logger.info(
                            "User opted out of destructive slash confirm (session=%s)",
                            session_key,
                        )
                    else:
                        logger.warning(
                            "Could not persist destructive_slash_confirm=false "
                            "(session=%s); config.yaml is not writable",
                            session_key,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist destructive_slash_confirm=false: %s", exc,
                    )
            result = await execute()
            if choice == "always":
                if persisted:
                    note = (
                        "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                        "without confirmation. Re-enable via "
                        "`approvals.destructive_slash_confirm: true` in config.yaml."
                    )
                else:
                    # The user did approve this run, so the action still goes
                    # ahead, but the preference did not stick and the prompt
                    # will be back next time. Say so rather than promising an
                    # opt-out that was never written.
                    note = (
                        "\n\n⚠️ Could not save that preference (config.yaml is not "
                        "writable), so /clear, /new, /reset, and /undo will ask "
                        "again next time. To silence it permanently, set "
                        "`approvals.destructive_slash_confirm: false` in config.yaml."
                    )
                if isinstance(result, str):
                    return result + note
                # EphemeralReply or other: leave untouched, since the note would
                # mangle structured replies.
                return result
            return result

        _p = self._typed_command_prefix_for(event.source.platform)
        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            f"_Text fallback: reply `{_p}approve`, `{_p}always`, or `{_p}cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _request_slash_confirm(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        message: str,
        handler,
    ) -> Optional[str]:
        """Ask the user to confirm an expensive slash command.

        ``handler`` is an async callable ``handler(choice: str) -> str``
        where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
        The handler runs on the event loop when the user responds; its
        return value is sent back as a gateway message.

        Returns a short acknowledgment string to send immediately (before
        the user's response).  If buttons rendered successfully the ack
        is ``None`` (buttons are self-explanatory); if we fell back to
        text the message itself IS the ack.
        """
        from tools import slash_confirm as _slash_confirm_mod

        source = event.source
        session_key = self._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError.  Real runs always
        # have the instance attribute.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self._adapter_for_source(source)
        metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

        used_buttons = False
        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id,
                    title=title,
                    message=message,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    used_buttons = True
            except Exception as exc:
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message

    def _read_user_config(self) -> Dict[str, Any]:
        """Read the user's raw config.yaml (cached) for gate lookups.

        Used by slash-confirm gates that must reflect on-disk state changes
        (e.g. a prior "Always Approve" click) without a gateway restart.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}
    @dataclasses.dataclass
    class _HygieneSettings:
        """Resolved session-hygiene configuration for one inbound turn."""
        model: str
        threshold_pct: float
        compression_enabled: bool
        hard_msg_limit: int
        timeout_seconds: float
        total_ceiling_seconds: float
        max_turn_hold_seconds: float
        failure_cooldown_seconds: float
        config_context_length: Optional[int]
        provider: Optional[str]
        base_url: Optional[str]
        api_key: Optional[str]
        data: Any

    @dataclasses.dataclass
    class _HygieneAttempt:
        """One detached hygiene compression attempt. ``cleanup_deferred`` is shared mutable state: wait
        handlers set it on raise paths; the owning ``finally`` reads it to decide on cleanup now."""
        agent: Any
        meta: Any
        commit_fence: Any = None
        future: Any = None
        wait_started: float = 0.0
        cleanup_deferred: bool = False
        history: Any = None

    def _thread_metadata_for_source(
        self, source, reply_to_message_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        metadata = self._thread_metadata_for_target(
            getattr(source, "platform", None), getattr(source, "chat_id", None),
            getattr(source, "thread_id", None), chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None))
        if getattr(source, "platform", None) == Platform.SLACK:
            # Per-turn egress identity: Slack chat.startStream needs recipient_user_id/team_id; the relay
            # adapter's _with_scope fallback reads per-chat caches a CONCURRENT turn overwrites.
            # Slack's chat.startStream requires recipient_user_id (+ recipient_team_id) when streaming to a
            # channel, and the relay connector fills those from metadata.user_id / metadata.scope_id. The
            # relay adapter's _with_scope fallback resolves BOTH from per-chat caches keyed only by chat_id
            # — mutable state that a CONCURRENT turn overwrites: two users with overlapping turns in one
            # channel would open U1's stream with U2 as the recipient. Stamp the authentic per-turn values
            # from THIS turn's source here, where they are still turn-scoped; _with_scope only fills keys
            # that are absent, so the cache degrades to what it should be — a restart/synthetic-send
            # fallback. See #210.
            team_id = getattr(source, "scope_id", None)
            user_id = getattr(source, "user_id", None)
            if team_id or user_id:
                metadata = dict(metadata or {})
                if team_id:
                    metadata["slack_team_id"] = str(team_id)
                    metadata.setdefault("scope_id", str(team_id))
                if user_id:
                    metadata.setdefault("user_id", str(user_id))
        from gateway.session_context import source_route_metadata
        metadata = source_route_metadata(source, metadata)
        # Routed profile for shared state.db namespaces: under profile_routes the transport adapter's
        # stamp is not the profile that wrote the binding (Telegram prune path needs it).
        # See #76423.
        profile = str(getattr(source, "profile", None) or "").strip()
        if profile and metadata is not None:
            metadata = dict(metadata)
            metadata["hermes_profile"] = profile
        return metadata

    def _thread_metadata_for_target(
        self, platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform, chat_id, thread_id, chat_type=chat_type, adapter=adapter):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # DM topic lanes need direct_messages_topic_id so synthetic sends reach the topic without a reply anchor.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        if platform == Platform.SLACK and reply_to_message_id is not None:
            # Slack's reply_in_thread=false path uses message_id to tell real threads from synthetic keys.
            metadata["message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, adapter: Optional[Any] = None) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Resolve the lookup on the CLASS, not the instance: getattr() on a MagicMock auto-creates callable
        # children, so an instance lookup would report a DM topic for every test double. Only a dict counts.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

    _reply_anchor_for_event = staticmethod(_reply_anchor_for_event)

    # Built-in platforms where ``/update`` is allowed (programmatic interfaces must not trigger updates).
    # Plugin-migrated platforms declare ``allow_update_command=True`` on their ``PlatformEntry`` instead.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.SLACK, Platform.WHATSAPP, Platform.SIGNAL, Platform.MATRIX,
        Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK, Platform.WEIXIN, Platform.BLUEBUBBLES, Platform.QQBOT, Platform.LOCAL,
    })

    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables (contextvars, not os.environ, so concurrent messages can't
        overwrite each other). Returns reset tokens for ``_clear_session_env`` in a ``finally``."""
        from gateway.session_context import set_session_vars
        # Async-delivery capability tells async tools whether this channel can wake a later turn. Default
        # True keeps CLI/unknown paths working; stateless adapters (api_server) declare False.
        _adapter = (getattr(self, "adapters", None) or {}).get(context.source.platform)
        _async_delivery = getattr(_adapter, "supports_async_delivery", True)
        return set_session_vars(
            platform=context.source.platform.value,
            chat_id=context.source.chat_id,
            chat_type=str(context.source.chat_type) if context.source.chat_type else "",
            chat_name=context.source.chat_name or "",
            thread_id=str(context.source.thread_id) if context.source.thread_id else "",
            user_id=str(context.source.user_id) if context.source.user_id else "",
            user_id_alt=str(context.source.user_id_alt) if context.source.user_id_alt else "",
            user_name=str(context.source.user_name) if context.source.user_name else "",
            scope_id=str(getattr(context.source, "scope_id", "") or ""),
            parent_chat_id=str(getattr(context.source, "parent_chat_id", "") or ""),
            session_key=context.session_key,
            message_id=str(context.source.message_id) if context.source.message_id else "",
            profile=getattr(context.source, "profile", "") or "",
            async_delivery=_async_delivery,
            cron_session="")

    def _clear_session_env(self, tokens: list) -> None:
        """Restore session context variables to their pre-handler values."""
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)

    async def _run_in_executor_with_context(self, func, *args):
        """Run blocking work in the thread pool while preserving session contextvars."""
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(self._get_executor(), ctx.run, func, *args)

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the gateway-owned executor for blocking agent work."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._executor_lock = lock
        with lock:
            if getattr(self, "_executor_closing", False):
                raise RuntimeError("Gateway is shutting down; executor unavailable")
            executor = getattr(self, "_executor", None)
            if executor is None or getattr(executor, "_shutdown", False):
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=10, thread_name_prefix="hermes-gateway")
                self._executor = executor
            return executor

    def _shutdown_executor(self, drain_timeout: float = 0.0) -> int:
        """Stop the gateway-owned executor; returns the number of worker threads still running.
        ``drain_timeout=0`` is fire-and-forget; shutdown passes a bounded budget so blocking DB work
        cannot outlive ``SessionDB.close()``. ``cancel_futures`` only drops unstarted work and cancelling
        a ``run_in_executor`` awaitable does not stop its thread, so running workers are joined."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            return 0
        with lock:
            self._executor_closing = True
            executor = getattr(self, "_executor", None)
            self._executor = None
        if executor is None:
            return 0
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

        # shutdown() has no timeout, so join workers directly; `_threads` is absent on test doubles (no wait).
        workers = list(getattr(executor, "_threads", None) or ())
        deadline = time.monotonic() + max(float(drain_timeout or 0.0), 0.0)
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)
        return sum(1 for worker in workers if worker.is_alive())

    # (section, key) config values baked into the agent at construction: a change MUST invalidate the
    # cached agent or a mid-gateway edit is silently ignored. Add new baked-in settings here.
    # _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"), ("compression", "enabled"),
        ("compression", "progress_notices"), ("compression", "threshold"),
        ("compression", "model_thresholds"), ("compression", "threshold_tokens"),
        ("compression", "codex_gpt55_autoraise"), ("compression", "codex_app_server_auto"),
        ("compression", "codex_responses_native"),
        ("compression", "codex_responses_compact_threshold"), ("compression", "in_place"),
        ("compression", "checkpoint_required"), ("compression", "micro_compact"),
        ("compression", "micro_compact_every_n_turns"),
        ("compression", "micro_compact_defrag_threshold_tokens"), ("compression", "target_ratio"),
        ("compression", "tail_mode"), ("compression", "protect_last_n"),
        ("compression", "proactive_prune_tokens"),
        ("compression", "proactive_prune_min_result_chars"),
        ("compression", "proactive_prune_min_reclaim_tokens"),
        ("compression", "min_tail_user_messages"), ("agent", "disabled_toolsets"),
        ("memory", "provider"), ("checkpoints", "enabled"), ("checkpoints", "max_snapshots"),
        ("checkpoints", "max_total_size_mb"), ("checkpoints", "max_file_size_mb"))

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.
        The activity ts/desc/provenance triple resets together and only at depth 0 — else a session idle
        29 min trips the watchdog before the first call; interrupt-recursive turns keep it so stuck-turn
        idle time accumulates to the 30-min timeout.

        ``_last_activity_ts``, ``_last_activity_desc``, and ``_last_activity_provenance`` are only reset for
        fresh external turns (depth 0); they are a semantic triple - description and provenance describe the
        activity *at* ts, so updating one without the others would make get_activity_summary() misleading.
        See #15654, #9051.
        """
        if interrupt_depth == 0:
            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
            agent._last_activity_provenance = ActivityProvenance.UNKNOWN
            # Reset the SessionDB flush cursor so the new turn's messages are fully persisted — a stale
            # value from the previous turn makes `_flush_messages_to_session_db` skip new rows.
            # See #44327.
            if hasattr(agent, "_last_flushed_db_idx"):
                agent._last_flushed_db_idx = 0
        agent._api_call_count = 0

    def _profile_name_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the profile name for an inbound source via configured routes (most specific wins).
        ``None`` = default/active profile. Gated on ``multiplex_profiles``, since the scoped run only
        activates under multiplexing; otherwise keys would be profile-namespaced while the agent ran in
        ``agent:main``."""
        config = getattr(self, "config", None)
        if not getattr(config, "multiplex_profiles", False):
            return None
        routes = getattr(config, "profile_routes", None)
        if not routes:
            return None
        from gateway.profile_routing import ProfileRouteRejected, match_profile_route
        try:
            matched = match_profile_route(
                routes, platform=source.platform.value, guild_id=getattr(source, "guild_id", None),
                chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
                parent_chat_id=getattr(source, "parent_chat_id", None))
        except Exception:
            logger.warning(
                "Profile route matching failed for %s/%s, falling back to default",
                source.platform, source.chat_id, exc_info=True)
            return None
        if matched:
            try:
                served = {name for name, _home in _multiplex_profile_homes(config)}
            except Exception as exc:
                logger.warning(
                    "Rejecting profile route %r because the served-profile set could not be resolved",
                    matched.name, exc_info=True)
                raise ProfileRouteRejected(matched.name) from exc
            if matched.profile not in served:
                logger.warning(
                    "Rejecting profile route %r: target profile %r is not served",
                    matched.name, matched.profile)
                raise ProfileRouteRejected(matched.name)
            return matched.profile
        logger.debug(
            "No profile route matched: platform=%s chat_id=%s thread_id=%s parent_chat_id=%s",
            source.platform.value, source.chat_id,
            getattr(source, "thread_id", None), getattr(source, "parent_chat_id", None))
        return None

    def _resolve_profile_home_for_source(self, source: SessionSource) -> "Path":
        """Resolve which profile's HERMES_HOME serves this source: ``source.profile``, then
        ``_profile_name_for_source`` (sources bypassing ``build_source``), then the active profile."""
        from gateway.profile_routing import ProfileRouteRejected
        from hermes_cli.profiles import get_active_profile_name, get_profile_dir, profile_exists
        from hermes_constants import get_hermes_home
        explicit_profile = None  # explicitly requested (source or routing) vs. default fallback
        try:
            name = (source.profile or "").strip() or self._profile_name_for_source(source)
            explicit_profile = name or None
            if not name:
                name = get_active_profile_name() or "default"
            profile_dir = get_profile_dir(name)
            if explicit_profile and not profile_exists(name):
                logger.warning(
                    "Profile %r does not exist for source %s/%s (guild_id=%s), "
                    "falling back to global HERMES_HOME",
                    explicit_profile, source.platform.value, source.chat_id,
                    getattr(source, "guild_id", None))
                return get_hermes_home()
            return profile_dir
        except ProfileRouteRejected:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve profile directory for source %s/%s (guild_id=%s), "
                "falling back to global HERMES_HOME: %s",
                source.platform.value, source.chat_id, getattr(source, "guild_id", None),
                explicit_profile or "(no profile)", exc_info=True)
            return get_hermes_home()

    async def _run_agent_inner(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.

        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool

        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        # ---- Proxy mode: delegate to remote API server ----
        if self._get_proxy_url():
            return await self._run_agent_via_proxy(
                message=message,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=event_message_id,
            )

        from run_agent import AIAgent
        import queue

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)

        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        enabled_toolsets = self._resolve_enabled_toolsets_for_source(
            user_config, source, platform_key
        )
        agent_cfg_local = user_config.get("agent") or {}
        from agent.skill_utils import parse_config_string_list

        disabled_toolsets = parse_config_string_list(agent_cfg_local.get("disabled_toolsets")) or None
        if (
            bool(getattr(source, "_voice_fast_lane", False))
            and not self._voice_fast_lane_requests_work(message)
        ):
            # No autonomous tool loop for conversational voice. An explicit
            # operation request retains the configured Discord toolsets and
            # every downstream approval/authorization guard.
            enabled_toolsets = []

        display_config = user_config.get("display", {})
        if not isinstance(display_config, dict):
            display_config = {}

        # Per-platform display settings — resolve via display_config module
        # which checks display.platforms.<platform>.<key> first, then
        # display.<key> global, then built-in platform defaults.
        from gateway.display_config import resolve_display_setting

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        # Apply friendly tool labels config (default on) — per-platform aware
        try:
            from agent.display import set_friendly_tool_labels
            _ftl = resolve_display_setting(user_config, platform_key, "friendly_tool_labels", True)
            set_friendly_tool_labels(bool(_ftl))
        except Exception:
            pass

        # Tool progress mode — resolved per-platform with env var fallback
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _display_cfg = display_config if isinstance(display_config, dict) else {}
        _platforms_cfg = _display_cfg.get("platforms") or {}
        _platform_cfg = _platforms_cfg.get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = (
            "tool_progress" in _display_cfg
            or (
                isinstance(_platform_cfg, dict)
                and "tool_progress" in _platform_cfg
            )
            or (
                isinstance(_legacy_tp_overrides, dict)
                and platform_key in _legacy_tp_overrides
            )
        )
        progress_mode = (
            _env_tp
            if _env_tp and not _tool_progress_configured
            else (_resolved_tp or _env_tp or "all")
        )
        # Tool progress grouping: "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
        _generic_status_recent: List[str] = []
        _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

        def _display_surface_mode(
            setting: str,
            *,
            default: bool = False,
            require_platform_override_for: set[Any] | None = None,
            allow_generic: bool = False,
        ) -> str:
            """Return off|raw|generic for a gateway visibility surface."""
            if require_platform_override_for:
                current_platform = _gateway_platform_value(source.platform)
                platform_only = {
                    _gateway_platform_value(item)
                    for item in require_platform_override_for
                }
                if (
                    current_platform in platform_only
                    and not _has_platform_display_override(user_config, platform_key, setting)
                ):
                    return "off"
            value = resolve_display_setting(user_config, platform_key, setting, default)
            if isinstance(value, str) and value.strip().lower() == "generic":
                return "generic" if allow_generic else "off"
            return "raw" if bool(value) else "off"

        def _generic_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None, args: Any = None) -> str:
            try:
                return choose_status_phrase(
                    kind,
                    tool_name=tool_name,
                    preview=preview,
                    args=args,
                    recent=_generic_status_recent,
                    catalog=_generic_status_catalog,
                )
            except Exception as _phrase_err:
                logger.debug("generic status phrase selection failed: %s", _phrase_err)
                return "still on it" if kind in {"heartbeat", "waiting", "long_running", "status"} else "one sec"
        # Disable tool progress for webhooks - they don't support message editing,
        # so each progress line would be sent as a separate message.
        from gateway.config import Platform
        tool_progress_enabled = progress_mode not in {"off", "log"} and source.platform != Platform.WEBHOOK
        # Live working-state status for text-rendering typing indicators
        # (Slack's assistant status line). Independent of tool_progress —
        # Slack defaults tool_progress off (permanent lines spam channels)
        # but the status line is ephemeral, so live status stays useful
        # there. Rendering rides the existing _keep_typing refresh: the
        # callback only stores a phrase on the adapter, costing zero extra
        # platform API calls.
        _live_status_mode = resolve_display_setting(
            user_config, platform_key, "live_status", "full"
        )
        _live_status_adapter = self._adapter_for_source(source)
        if not getattr(_live_status_adapter, "supports_status_text", False):
            _live_status_adapter = None
        if _live_status_mode == "off":
            _live_status_adapter = None
        # "log" mode: tool calls are written to ~/.hermes/logs/tool_calls.log
        # instead of the chat (#3459 / #3458). Gateway-only by design.
        log_mode_enabled = progress_mode == "log" and source.platform != Platform.WEBHOOK
        log_queue: "queue.Queue | None" = queue.Queue() if log_mode_enabled else None
        # Natural assistant status messages are intentionally independent from
        # tool progress and token streaming. Users can keep tool_progress quiet
        # in chat platforms while opting into concise mid-turn updates.
        interim_assistant_messages_mode = _display_surface_mode(
            "interim_assistant_messages",
            default=True,
            require_platform_override_for={Platform.MATTERMOST},
        )
        interim_assistant_messages_enabled = (
            source.platform != Platform.WEBHOOK
            and interim_assistant_messages_mode != "off"
        )
        # thinking_progress is independent — if enabled, we need the progress
        # queue even when tool_progress is off (thinking relay uses same infra).
        # Mattermost requires a per-platform opt-in: global scratch-text display
        # is too easy to leak into busy public threads.
        _thinking_mode = _display_surface_mode(
            "thinking_progress",
            default=False,
            require_platform_override_for={Platform.MATTERMOST},
        )
        _thinking_enabled = _thinking_mode != "off"
        # Slack-native task cards (#29483): when the Slack adapter's opt-in
        # is set, tool progress renders as native plan/task cards via
        # chat.startStream — the progress queue is needed even though Slack
        # keeps ordinary text tool_progress off by default (requiring both
        # flags would silently leave the native feature inactive).
        _progress_adapter_for_native = self._adapter_for_source(source)
        _native_slack_task_cards = False
        if (
            source.platform == Platform.SLACK
            and _progress_adapter_for_native is not None
            and hasattr(_progress_adapter_for_native, "native_task_cards_enabled")
        ):
            try:
                _native_slack_task_cards = bool(
                    _progress_adapter_for_native.native_task_cards_enabled()
                )
            except Exception:
                logger.debug("Slack native task-card config check failed", exc_info=True)
        needs_progress_queue = (
            tool_progress_enabled or _thinking_enabled or _native_slack_task_cards
        )


        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if needs_progress_queue else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated
        # True when the previously enqueued progress line was a terminal
        # fenced code block — consecutive terminal calls then drop the
        # repeated "💻 terminal" header and render back-to-back blocks.
        last_was_terminal_block = [False]

        # ── Discord voice "verbal ack before tool calls" ────────────────
        # When the bot is in a voice channel with the continuous mixer
        # installed (discord.voice_fx.enabled), speak a short phrase ("let me
        # look into that") over the ambient idle bed on the FIRST tool call of
        # the turn.  Fires from tool_start_callback (independent of the
        # tool-progress text gate), at most once per turn.  No-op on every
        # other platform / when not in a voice channel.
        _voice_ack_fired = [False]
        _voice_ack_guild: List[Optional[int]] = [None]
        if source.platform == Platform.DISCORD:
            _va = self.adapters.get(Platform.DISCORD)
            # source.chat_id is the linked text channel; resolve the guild whose
            # voice connection is bound to it (mirrors DiscordAdapter.play_tts).
            _vtc = getattr(_va, "_voice_text_channels", None)
            if isinstance(_vtc, dict) and hasattr(_va, "voice_mixer_active"):
                for _gid, _tc in _vtc.items():
                    if str(_tc) == str(source.chat_id) and _va.voice_mixer_active(_gid):
                        _voice_ack_guild[0] = _gid
                        break
        _voice_ack_loop = asyncio.get_running_loop()

        # voice_ack_callback extracted to TurnRunner.voice_ack_callback
        # (published onto turn_ctx after the runner is constructed below).

        # Auto-cleanup of temporary progress bubbles (Telegram + any adapter
        # that implements ``delete_message``). When enabled via
        # ``display.platforms.<platform>.cleanup_progress: true``, message IDs
        # from the tool-progress / "⏳ Working — N min" / status-callback bubbles
        # are collected here and deleted after the final response lands.
        # Failed runs skip cleanup so the bubbles remain as breadcrumbs.
        _cleanup_progress = bool(
            resolve_display_setting(user_config, platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self._adapter_for_source(source) if _cleanup_progress else None
        # getattr, not attribute access — same duck-typed-adapter guard as the
        # edit_message check in send_progress_messages below: a fake/minimal
        # adapter without delete_message means "can't delete", not a crash.
        _cleanup_delete = getattr(type(_cleanup_adapter), "delete_message", None) if _cleanup_adapter is not None else None
        if _cleanup_adapter is not None and (
            _cleanup_delete is None
            or _cleanup_delete is BasePlatformAdapter.delete_message
        ):
            # Adapter doesn't support deletion — silently disable.
            _cleanup_progress = False
            _cleanup_adapter = None
        _cleanup_msg_ids: List[str] = []
        # First-touch onboarding latch: fires at most once per run, even if
        # several tools exceed the threshold.
        long_tool_hint_fired = [False]
        _LONG_TOOL_THRESHOLD_S = 30.0

        turn_ctx = TurnContext(
            source=source,
            _run_still_current=_run_still_current,
            _live_status_adapter=_live_status_adapter,
            _live_status_mode=_live_status_mode,
            _thinking_enabled=_thinking_enabled,
            progress_mode=progress_mode,
            progress_grouping=progress_grouping,
            tool_progress_enabled=tool_progress_enabled,
            progress_queue=progress_queue,
            log_queue=log_queue,
            last_progress_msg=last_progress_msg,
            last_tool=last_tool,
            last_was_terminal_block=last_was_terminal_block,
            repeat_count=repeat_count,
            long_tool_hint_fired=long_tool_hint_fired,
            _LONG_TOOL_THRESHOLD_S=_LONG_TOOL_THRESHOLD_S,
            _cleanup_progress=_cleanup_progress,
            _cleanup_msg_ids=_cleanup_msg_ids,
            message=message,
            AIAgent=AIAgent,
            resolve_display_setting=resolve_display_setting,
            user_config=user_config,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            log_mode_enabled=log_mode_enabled,
            interim_assistant_messages_enabled=interim_assistant_messages_enabled,
            needs_progress_queue=needs_progress_queue,
            _native_slack_task_cards=_native_slack_task_cards,
            _voice_ack_fired=_voice_ack_fired,
            _voice_ack_guild=_voice_ack_guild,
            _voice_ack_loop=_voice_ack_loop,
            history=history,
            context_prompt=context_prompt,
            channel_prompt=channel_prompt,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id,
            moa_config=moa_config,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
        )
        turn_runner = TurnRunner(self, turn_ctx)
        # Callback invoked by agent on tool lifecycle events — extracted to
        # TurnRunner.progress_callback (bound method, same signature).
        turn_ctx.progress_callback = turn_runner.progress_callback
        turn_ctx.voice_ack_callback = turn_runner.voice_ack_callback
        turn_ctx.native_tool_start_callback = turn_runner.combined_tool_start_callback
        turn_ctx.native_tool_complete_callback = (
            turn_runner.native_tool_complete_callback
        )

        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited.
        #
        # Threading metadata is platform-specific:
        # - Slack DM threading needs event_message_id fallback (reply thread)
        # - Telegram forum topics use message_thread_id; Hermes-created private
        #   DM topic lanes require both thread metadata and a reply anchor
        # - Feishu only honors reply_in_thread when sending a reply, so topic
        #   progress uses the triggering event message as the reply target
        # - Other platforms should use explicit source.thread_id only
        #
        # Slack honours platforms.slack.extra.reply_in_thread=false: if the
        # user has opted out of threaded replies, don't synthesise a thread
        # for progress messages either — the very first progress message
        # would otherwise create a thread that all subsequent replies
        # (including the final answer) would inherit (#18859).
        _progress_reply_in_thread = True
        if source.platform == Platform.SLACK:
            _slack_adapter_for_progress = self._adapter_for_source(source)
            if _slack_adapter_for_progress is not None:
                try:
                    # Relay lane: the adapter owns mode resolution (nested
                    # platforms.relay.extra.slack subset with flat-key
                    # fallback). Native lane: read the flat extra as before.
                    _mode_fn = getattr(
                        _slack_adapter_for_progress,
                        "_effective_reply_in_thread",
                        None,
                    )
                    if callable(_mode_fn):
                        _progress_reply_in_thread = bool(_mode_fn())
                    else:
                        _progress_reply_in_thread = bool(
                            _slack_adapter_for_progress.config.extra.get(
                                "reply_in_thread", True
                            )
                        )
                except Exception:
                    _progress_reply_in_thread = True
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id,
            reply_in_thread=_progress_reply_in_thread,
        )
        # Relay Discord auto-thread lane: a channel-initiating message has no
        # thread_id at ingest (the thread is born on the connector's FIRST
        # send). The connector stamps prospective_thread_id (the anchor message
        # id, == the id of the thread it will create) and auto-threads any
        # outbound carrying that anchor as reply_to. Without it, the progress /
        # tool-status bubble is sent flat (no thread, no anchor) and lands in
        # the PARENT channel while the final reply threads — the search-status
        # updates leaked outside the thread (staging repro 2026-08-02). Carry
        # the anchor on the progress send so it routes into the SAME auto-thread.
        _relay_prospective_thread_id = (
            str(getattr(source, "prospective_thread_id", None))
            if source.platform == Platform.DISCORD
            and getattr(source, "delivered_via_upstream_relay", False)
            and getattr(source, "prospective_thread_id", None)
            and not source.thread_id
            else None
        )
        _progress_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else self._thread_metadata_for_target(
                source.platform,
                source.chat_id,
                _progress_thread_id,
                chat_type=getattr(source, "chat_type", None),
                reply_to_message_id=event_message_id,
            )
        ) if _progress_thread_id else None
        if _progress_metadata is None and _relay_prospective_thread_id:
            # No real thread yet, but the connector will auto-thread on the
            # reply anchor; carry it so progress joins that thread.
            _progress_metadata = {"reply_to_message_id": event_message_id}
        _progress_metadata = _non_conversational_metadata(_progress_metadata, platform=source.platform)
        if _native_slack_task_cards:
            # chat.startStream in channels requires the recipient team/user
            # pair; harmless extras elsewhere, so stamp them whenever known.
            _progress_metadata = dict(_progress_metadata or {})
            if source.scope_id:
                _progress_metadata.setdefault("recipient_team_id", source.scope_id)
                _progress_metadata.setdefault("slack_team_id", source.scope_id)
            if source.user_id:
                _progress_metadata.setdefault("recipient_user_id", source.user_id)
        _progress_reply_to = (
            event_message_id
            if (
                source.platform in (Platform.FEISHU, Platform.MATTERMOST)
                and source.thread_id
                and event_message_id
            )
            or _relay_prospective_thread_id
            else None
        )

        async def write_tool_log():
            """Drain log_queue and append tool-call lines to tool_calls.log.

            Only active when ``display.tool_progress`` is ``log``. Uses a
            RotatingFileHandler (5MB × 3 backups) so the audit log can't grow
            unbounded, and the shared RedactingFormatter so secrets never land
            on disk.
            """
            if log_queue is None:
                return
            from logging.handlers import RotatingFileHandler

            from agent.redact import RedactingFormatter

            log_dir = _hermes_home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "tool_calls.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(RedactingFormatter("%(message)s"))
            tool_logger = logging.getLogger(f"hermes.tool_calls.{id(log_queue)}")
            tool_logger.setLevel(logging.INFO)
            tool_logger.propagate = False
            tool_logger.addHandler(file_handler)
            try:
                while True:
                    try:
                        tool_logger.info("%s", log_queue.get_nowait())
                    except queue.Empty:
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.error("write_tool_log error: %s", e)
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                # Drain remaining entries before closing so late tool calls
                # from the final iteration aren't lost.
                while True:
                    try:
                        tool_logger.info("%s", log_queue.get_nowait())
                    except queue.Empty:
                        break
                    except Exception:
                        break
                tool_logger.removeHandler(file_handler)
                try:
                    file_handler.flush()
                    file_handler.close()
                except Exception:
                    pass

        # Extracted to TurnRunner.send_progress_messages. The threading
        # metadata computed above is published onto the shared TurnContext
        # exactly where the original closure's captured locals were bound.
        turn_ctx._progress_metadata = _progress_metadata
        turn_ctx._progress_reply_to = _progress_reply_to
        send_progress_messages = turn_runner.send_progress_messages

        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        turn_ctx.agent_holder = agent_holder
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        # #60671 — streaming PCM audio consumer.  Created on the gateway
        # event-loop thread (NOT inside run_sync's executor worker) so the
        # outer finalisation / interrupt paths can reference it without a
        # cross-scope NameError.
        streaming_tts_consumer_holder: list = [None]
        turn_ctx.result_holder = result_holder
        turn_ctx.tools_holder = tools_holder
        turn_ctx.stream_consumer_holder = stream_consumer_holder
        turn_ctx.streaming_tts_consumer_holder = streaming_tts_consumer_holder

        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_running_loop()
        _hooks_ref = self.hooks

        # Bridge extracted to TurnRunner._step_callback_sync; the loop and
        # hooks refs bound just above are published at their original site.
        turn_ctx._loop_for_step = _loop_for_step
        turn_ctx._hooks_ref = _hooks_ref
        turn_ctx._step_callback_sync = turn_runner._step_callback_sync

        # Bridge sync event_callback → async hooks.emit for lifecycle events
        # (e.g. session:compress fires after context compression splits a session)
        # Bridge extracted to TurnRunner._event_callback_sync.
        turn_ctx._event_callback_sync = turn_runner._event_callback_sync

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self._adapter_for_source(source)
        _status_chat_id = source.chat_id
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            # Feishu topics only keep messages inside the topic when they are
            # sent via the reply API with reply_in_thread=true. Status/interim,
            # approval, and stream-consumer paths usually only receive metadata,
            # so carry the triggering message id as a Feishu-specific fallback.
            _status_thread_metadata: Optional[Dict[str, Any]] = {
                "thread_id": _progress_thread_id,
                "reply_to_message_id": event_message_id,
            }
        else:
            _status_thread_metadata = (
                self._thread_metadata_for_source(source, event_message_id)
                if _progress_thread_id == source.thread_id
                else self._thread_metadata_for_target(
                    source.platform,
                    source.chat_id,
                    _progress_thread_id,
                    chat_type=getattr(source, "chat_type", None),
                    reply_to_message_id=event_message_id,
                )
            ) if _progress_thread_id else None
            if _status_thread_metadata is None and _relay_prospective_thread_id:
                # Relay Discord auto-thread lane (see _progress_metadata above):
                # carry the reply anchor so status/interim bubbles route into
                # the same connector-created thread as the final reply.
                _status_thread_metadata = {
                    "reply_to_message_id": event_message_id
                }

        # Bridge extracted to TurnRunner._status_callback_sync; publish the
        # status wiring computed above onto the shared TurnContext at the
        # exact original binding site.
        turn_ctx._status_adapter = _status_adapter
        turn_ctx._status_chat_id = _status_chat_id
        turn_ctx._status_thread_metadata = _status_thread_metadata
        turn_ctx._status_callback_sync = turn_runner._status_callback_sync

        # ---- Streaming TTS consumer setup (#60671) ----
        # Created on the gateway event-loop thread (here, in _run_agent_inner),
        # NOT inside run_sync's executor worker.  This avoids a cross-scope
        # NameError: the outer interrupt / finalisation paths reference the
        # consumer via ``streaming_tts_consumer_holder[0]``.
        #
        # Gates: voice input, auto-TTS enabled for this chat, adapter
        # supports streaming, and a usable streaming TTS provider configured.
        _stts_adapter = self._adapter_for_source(source)
        _is_voice_input = (
            message_type is not None
            and str(getattr(message_type, "value", message_type)).lower() == "voice"
        )
        if (
            _stts_adapter is not None
            and _is_voice_input
            and _stts_adapter._should_auto_tts_for_chat(source.chat_id)
        ):
            try:
                from gateway.streaming_tts_consumer import StreamingTTSConsumer
                from tools.tts_tool import _load_tts_config
                _tts_cfg = _load_tts_config()
                _gateway_loop = self._gateway_loop or asyncio.get_event_loop()
                _stts_consumer = StreamingTTSConsumer(
                    adapter=_stts_adapter,
                    chat_id=source.chat_id,
                    tts_config=_tts_cfg,
                    loop=_gateway_loop,
                    metadata=_status_thread_metadata,
                )
                if _stts_consumer.active:
                    streaming_tts_consumer_holder[0] = _stts_consumer
                    _stts_consumer.start()
                # else: consumer inactive (no streaming provider) — leave
                # the holder as None so the whole-file fallback path runs.
            except Exception as _stts_err:
                logger.debug("Could not set up streaming TTS consumer: %s", _stts_err)

        # run_sync extracted to TurnRunner.run_sync (bound method; the
        # executor call below is unchanged).  Its closed-over locals travel
        # on turn_ctx; `nonlocal message` rebinds became ctx.message writes.
        run_sync = turn_runner.run_sync

        # Start progress message sender if enabled. Gate on needs_progress_queue
        # (tool_progress OR thinking_progress), not tool_progress alone: the
        # sender drains BOTH tool-progress lines and _thinking scratch bubbles.
        # With the old tool_progress-only gate, a thinking_progress:true /
        # tool_progress:off user had the callback queue _thinking messages that
        # no task ever drained — so they silently never appeared.
        progress_task = None
        if needs_progress_queue:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start the tool-call log writer when tool_progress == "log".
        log_task = None
        if log_mode_enabled:
            log_task = asyncio.create_task(write_tool_log())

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None

        async def _start_stream_consumer():
            """Wait for the stream consumer to be created, then run it."""
            for _ in range(200):  # Up to 10s wait
                if stream_consumer_holder[0] is not None:
                    await stream_consumer_holder[0].run()
                    return
                await asyncio.sleep(0.05)

        stream_task = asyncio.create_task(_start_stream_consumer())

        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if not session_key:
                return
            # Only promote the sentinel to the real agent if this run is still
            # current.  If /stop or /new bumped the generation while we were
            # spinning up, leave the newer run's slot alone — we'll be
            # discarded by the stale-result check in _handle_message_with_agent.
            if run_generation is not None and not self._is_session_run_current(
                session_key, run_generation
            ):
                logger.info(
                    "Skipping stale agent promotion for %s — generation %s is no longer current",
                    session_key or "",
                    run_generation,
                )
                return
            self._session_state(session_key).turn.agent = agent_holder[0]
            if self._draining:
                self._update_runtime_status("draining")

        tracking_task = asyncio.create_task(track_agent())

        # Monitor for interrupts from the adapter (new messages arriving).
        # This is the PRIMARY interrupt path for regular text messages —
        # Level 1 (base.py) catches them before _handle_message() is reached,
        # so the Level 2 running_agent.interrupt() path never fires.
        # The inactivity poll loop below has a BACKUP check in case this
        # task dies (no error handling = silent death = lost interrupts).
        _interrupt_detected = asyncio.Event()  # shared with backup check

        async def monitor_for_interrupt():
            if not session_key:
                return

            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                try:
                    # Re-resolve adapter each iteration so reconnects don't
                    # leave us holding a stale reference.
                    _adapter = self._adapter_for_source(source)
                    if not _adapter:
                        continue
                    # Check if adapter has a pending interrupt for this session.
                    # Must use session_key (build_session_key output) — NOT
                    # source.chat_id — because the adapter stores interrupt events
                    # under the full session key.
                    if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                        agent = agent_holder[0]
                        if agent:
                            # Peek at the pending message text WITHOUT consuming it.
                            # The message must remain in _pending_messages so the
                            # post-run dequeue at _dequeue_pending_event() can
                            # retrieve the full MessageEvent (with media metadata).
                            # If we pop here, a race exists: the agent may finish
                            # before checking _interrupt_requested, and the message
                            # is lost — neither the interrupt path nor the dequeue
                            # path finds it.
                            _peek_event = _adapter._pending_messages.get(session_key)
                            pending_text = None
                            if _peek_event is not None:
                                pending_text = _peek_event.text or ""
                                # Transcribe audio media BEFORE signaling the
                                # agent, so voice messages interrupt with the
                                # real transcript instead of an empty string
                                # (or file-path placeholder). Matches the UX
                                # of fresh voice messages including the
                                # optional 🎙️ echo back to the user.
                                _media_urls = getattr(_peek_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_peek_event):
                                    pending_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _peek_event,
                                        _adapter,
                                        source,
                                        pending_text,
                                        log_context="Voice-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not pending_text and _media_urls:
                                    pending_text = _build_media_placeholder(_peek_event)
                            logger.debug("Interrupt detected from adapter, signaling agent...")
                            agent.interrupt(pending_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as _mon_err:
                    logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)

        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())

        # Periodic "still working" notifications for long-running tasks.
        # Fires every N seconds so the user knows the agent hasn't died.
        # Config: agent.gateway_notify_interval in config.yaml, or
        # HERMES_AGENT_NOTIFY_INTERVAL env var.  Default 180s (3 min).
        # 0 = disable notifications.
        _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
        _long_running_mode = _display_surface_mode(
            "long_running_notifications",
            default=True,
            allow_generic=True,
        )
        if _long_running_mode == "off":
            _NOTIFY_INTERVAL = None
        _notify_start = time.time()

        async def _notify_long_running():
            if _NOTIFY_INTERVAL is None:
                return  # Notifications disabled (gateway_notify_interval: 0)
            _notify_adapter = self._adapter_for_source(source)
            if not _notify_adapter:
                return
            # Track the heartbeat message id so we can edit-in-place on
            # platforms that support it (Telegram, Discord, Slack, etc.)
            # instead of spamming a new "Still working" bubble every
            # interval. Falls back to send-new when edit fails or isn't
            # supported by the adapter.
            _heartbeat_msg_id: Optional[str] = None
            while True:
                await asyncio.sleep(_NOTIFY_INTERVAL)
                # Stop heartbeating once this run no longer owns the session
                # slot or the executor has finished — otherwise a stale
                # "running: delegate_task" bubble can outlive the run that
                # spawned it (#12029). _executor_task is a closure var bound
                # just after this task is scheduled; tolerate the brief window
                # before then (the first wake is _NOTIFY_INTERVAL away anyway).
                try:
                    _exec_ref = _executor_task
                except NameError:
                    _exec_ref = None
                if not self._should_emit_long_running_notification(
                    session_key, agent_holder[0], _exec_ref
                ):
                    break
                _elapsed_mins = int((time.time() - _notify_start) // 60)
                # Include agent activity context if available. Default
                # heartbeat is terse: elapsed + current tool. Verbose
                # iteration counter is gated on busy_ack_detail so users
                # who want it can opt in per platform.
                _agent_ref = agent_holder[0]
                _status_detail = ""
                _want_iteration_detail = bool(
                    resolve_display_setting(
                        user_config,
                        platform_key,
                        "busy_ack_detail",
                        True,
                    )
                )
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _a = _agent_ref.get_activity_summary()
                        _parts = []
                        if _want_iteration_detail:
                            _parts.append(
                                f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                            )
                        _action = _a.get("current_tool") or _a.get("last_activity_desc")
                        if _action:
                            _parts.append(str(_action))
                        if _parts:
                            _status_detail = " — " + ", ".join(_parts)
                    except Exception:
                        pass
                _heartbeat_text = (
                    _generic_status_phrase("status")
                    if _long_running_mode == "generic"
                    else f"⏳ Working — {_elapsed_mins} min{_status_detail}"
                )
                try:
                    _notify_res = None
                    if _heartbeat_msg_id:
                        try:
                            _notify_res = await _notify_adapter.edit_message(
                                source.chat_id,
                                _heartbeat_msg_id,
                                _heartbeat_text,
                            )
                        except Exception as _ee:
                            logger.debug("Heartbeat edit failed: %s", _ee)
                            _notify_res = None
                    if not (_notify_res and getattr(_notify_res, "success", False)):
                        _notify_res = await _notify_adapter.send(
                            source.chat_id,
                            _heartbeat_text,
                            metadata=_interim_metadata(_non_conversational_metadata(_status_thread_metadata, platform=source.platform)),
                        )
                        if getattr(_notify_res, "success", False) and getattr(
                            _notify_res, "message_id", None
                        ):
                            _heartbeat_msg_id = str(_notify_res.message_id)
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(_heartbeat_msg_id)
                except Exception as _ne:
                    logger.debug("Long-running notification error: %s", _ne)

        _notify_task = asyncio.create_task(_notify_long_running())

        def _stream_confirmed_final_delivery(
            consumer,
            final_text: str,
            *,
            previewed: bool = False,
        ) -> bool:
            """Return True only when the actual final reply reached the user."""
            if consumer is None:
                return False
            if getattr(consumer, "final_response_sent", False):
                # A successful finalize call is not proof the *content* was
                # final: the edit may have carried only the last preview
                # snapshot while the tail generated between that snapshot and
                # stream completion never reached any API call (#71643).
                # Reconcile the recorded turn-final payload against the
                # completed response; only a demonstrable mismatch (False)
                # overrides the flag — including payload-less multi-message
                # split delivery (#78541). None (no record on a non-split
                # legacy path) keeps the legacy trust so ambiguous-timeout
                # dedup is not regressed.
                matcher = getattr(consumer, "delivered_final_matches", None)
                if callable(matcher):
                    try:
                        if matcher(final_text) is False:
                            return False
                    except Exception:
                        pass
                return True
            if previewed:
                has_delivered_text = getattr(consumer, "has_delivered_text", None)
                if callable(has_delivered_text):
                    try:
                        return bool(has_delivered_text(final_text))
                    except Exception:
                        return False
            return False

        try:
            # Run in thread pool to not block.  Use an *inactivity*-based
            # timeout instead of a wall-clock limit: the agent can run for
            # hours if it's actively calling tools / receiving stream tokens,
            # but a hung API call or stuck tool with no activity for the
            # configured duration is caught and killed.  (#4815)
            #
            # Config: agent.gateway_timeout in config.yaml, or
            # HERMES_AGENT_TIMEOUT env var (env var takes precedence).
            # Default 1800s (30 min inactivity).  0 = unlimited.
            _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
            _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
            _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
            _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None
            _warning_fired = False

            # A background=true process intentionally survives a successful
            # turn, so capture existing IDs and reap only children created by
            # THIS turn if it times out. The daemon watchdog is independent of
            # asyncio: cgroup memory reclaim may starve the event loop that runs
            # the normal timeout poll, but it need not also postpone cleanup
            # until the loop recovers (#76115).
            from tools.process_registry import process_registry

            _turn_task_id = session_id or ""
            _turn_process_baseline = process_registry.snapshot_running_ids(_turn_task_id)
            turn_ctx.process_task_id = _turn_task_id
            turn_ctx.process_baseline = _turn_process_baseline
            _turn_worker_done = threading.Event()
            _turn_timeout_fired = threading.Event()
            _turn_cleanup_lock = threading.Lock()
            # task_id above is session-scoped, not turn-scoped (#76115
            # review): gate the eventual reap on this exact claim still
            # being current, so a replacement turn that starts on the same
            # session before the watchdog fires doesn't get its own fresh
            # process killed by this turn's stale baseline.
            _turn_run_generation = run_generation
            _turn_is_current = (
                (lambda: self._is_session_run_current(session_key, _turn_run_generation))
                if _turn_run_generation is not None
                else (lambda: True)
            )

            def _run_sync_with_timeout_lifecycle():
                try:
                    return run_sync()
                finally:
                    _turn_worker_done.set()
                    # `.turn.agent` on the session state is only reset to
                    # _AGENT_PENDING_SENTINEL when the *next* turn is
                    # claimed (see _session_state(...).turn.agent = ... at
                    # claim time), so a stale reference to this exact agent
                    # instance stays reachable from
                    # _interrupt_and_clear_session() until then. Clearing
                    # the ownership markers here — the instant this turn's
                    # own worker finishes — closes that window: an
                    # explicit /stop landing on the already-finished turn
                    # no longer reaps background work the turn deliberately
                    # left running (#76115).
                    _finished_agent = agent_holder[0] if agent_holder else None
                    if _finished_agent is not None:
                        _finished_agent._gateway_turn_process_task_id = ""
                        _finished_agent._gateway_turn_process_baseline = frozenset()

            if _agent_timeout is not None:
                threading.Thread(
                    target=_watch_gateway_turn_inactivity,
                    kwargs={
                        "agent_holder": agent_holder,
                        "task_id": _turn_task_id,
                        "process_baseline": _turn_process_baseline,
                        "timeout": _agent_timeout,
                        "worker_done": _turn_worker_done,
                        "timeout_fired": _turn_timeout_fired,
                        "cleanup_lock": _turn_cleanup_lock,
                        "poll_interval": 5.0,
                        "is_still_current": _turn_is_current,
                    },
                    name=f"gateway-turn-watchdog-{_turn_task_id[:12]}",
                    daemon=True,
                ).start()
            _executor_task = asyncio.ensure_future(
                self._run_in_executor_with_context(_run_sync_with_timeout_lifecycle)
            )

            _inactivity_timeout = False
            _POLL_INTERVAL = 5.0

            if _agent_timeout is None:
                # Unlimited — still poll periodically for backup interrupt
                # detection in case monitor_for_interrupt() silently died.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Backup interrupt check: if the monitor task died or
                    # missed the interrupt, catch it here.
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self._adapter_for_source(source)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            if _bp_event is not None:
                                _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_bp_event):
                                    _bp_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _bp_event,
                                        _backup_adapter,
                                        source,
                                        _bp_text or "",
                                        log_context="Voice-backup-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not _bp_text and _bp_media_urls:
                                    _bp_text = _build_media_placeholder(_bp_event)
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")

            else:
                # Poll loop: check the agent's built-in activity tracker
                # (updated by _touch_activity() on every tool call, API
                # call, and stream delta) every few seconds.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        # Prefer the real result when the worker finished,
                        # even if the watchdog fired in the same window: the
                        # completed run already persisted its reply to session
                        # history, so surfacing the "agent inactive" diagnostic
                        # here would contradict the stored transcript. This
                        # mirrors _abandon_timed_out_gateway_turn's own
                        # worker_done-wins tiebreak (under cleanup_lock).
                        response = _executor_task.result()
                        break
                    if _turn_timeout_fired.is_set():
                        _inactivity_timeout = True
                        break
                    # Agent still running — check inactivity.
                    _agent_ref = agent_holder[0]
                    _idle_secs = 0.0
                    if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                        try:
                            _act = _agent_ref.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    # Staged warning: fire once before escalating to full timeout.
                    if (not _warning_fired and _agent_warning is not None
                            and _idle_secs >= _agent_warning):
                        _warning_fired = True
                        _warn_adapter = self._adapter_for_source(source)
                        if _warn_adapter:
                            _elapsed_warn = int(_agent_warning // 60) or 1
                            _remaining_mins = int((_agent_timeout - _agent_warning) // 60) or 1
                            try:
                                await _warn_adapter.send(
                                    source.chat_id,
                                    f"⚠️ No activity for {_elapsed_warn} min. "
                                    f"If the agent does not respond soon, it will "
                                    f"be timed out in {_remaining_mins} min. "
                                    f"You can continue waiting or use /reset.",
                                    metadata=_interim_metadata(_status_thread_metadata),
                                )
                            except Exception as _warn_err:
                                logger.debug("Inactivity warning send error: %s", _warn_err)
                    if _idle_secs >= _agent_timeout:
                        _inactivity_timeout = True
                        threading.Thread(
                            target=_abandon_timed_out_gateway_turn,
                            kwargs={
                                "agent_holder": agent_holder,
                                "task_id": _turn_task_id,
                                "process_baseline": _turn_process_baseline,
                                "worker_done": _turn_worker_done,
                                "timeout_fired": _turn_timeout_fired,
                                "cleanup_lock": _turn_cleanup_lock,
                                "is_still_current": _turn_is_current,
                            },
                            name=f"gateway-turn-reaper-{_turn_task_id[:12]}",
                            daemon=True,
                        ).start()
                        break
                    # Backup interrupt check (same as unlimited path).
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self._adapter_for_source(source)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            if _bp_event is not None:
                                _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_bp_event):
                                    _bp_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _bp_event,
                                        _backup_adapter,
                                        source,
                                        _bp_text or "",
                                        log_context="Voice-backup-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not _bp_text and _bp_media_urls:
                                    _bp_text = _build_media_placeholder(_bp_event)
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")

            if _inactivity_timeout:
                # Build a diagnostic summary from the agent's activity tracker.
                _timed_out_agent = agent_holder[0]
                _activity = {}
                if _timed_out_agent and hasattr(_timed_out_agent, "get_activity_summary"):
                    try:
                        _activity = _timed_out_agent.get_activity_summary()
                    except Exception:
                        pass

                _last_desc = _activity.get("last_activity_desc", "unknown")
                _secs_ago = _activity.get("seconds_since_activity", 0)
                _cur_tool = _activity.get("current_tool")
                _iter_n = _activity.get("api_call_count", 0)
                _iter_max = _activity.get("max_iterations", 0)

                logger.error(
                    "Agent idle for %.0fs (timeout %.0fs) in session %s "
                    "| last_activity=%s | iteration=%s/%s | tool=%s",
                    _secs_ago, _agent_timeout, session_key,
                    _last_desc, _iter_n, _iter_max,
                    _cur_tool or "none",
                )

                # Interrupt the agent if it's still running so the thread
                # pool worker is freed.
                if _timed_out_agent:
                    request_hard_interrupt(_timed_out_agent, _INTERRUPT_REASON_TIMEOUT)

                _timeout_mins = int(_agent_timeout // 60) or 1

                # Construct a user-facing message with diagnostic context.
                _diag_lines = [
                    f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                    f"or API responses."
                ]
                if _cur_tool:
                    _diag_lines.append(
                        f"The agent appears stuck on tool `{_cur_tool}` "
                        f"({_secs_ago:.0f}s since last activity, "
                        f"iteration {_iter_n}/{_iter_max})."
                    )
                else:
                    _diag_lines.append(
                        f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                        f"iteration {_iter_n}/{_iter_max}). "
                        "The agent may have been waiting on an API response."
                    )
                _diag_lines.append(
                    "To increase the limit, set agent.gateway_timeout in config.yaml "
                    "(value in seconds, 0 = no limit) and restart the gateway.\n"
                    "Try again, or use /reset to start fresh."
                )

                response = {
                    "final_response": "\n".join(_diag_lines),
                    "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                    "api_calls": _iter_n,
                    "tools": tools_holder[0] or [],
                    "history_offset": 0,
                    "failed": True,
                }

            # Track fallback model state: if the agent switched to a
            # fallback model during this run, persist it so /model shows
            # the actually-active model instead of the config default.
            # Skip eviction when the run failed — evicting a failed agent
            # forces MCP reinit on the next message for no benefit (the
            # same error will recur).  This was the root cause of #7130:
            # a bad model ID triggered fallback → eviction → recreation →
            # MCP reinit → same 400 → loop, burning 91% CPU for hours.
            _agent = agent_holder[0]
            _result_for_fb = result_holder[0]
            _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
            if _agent is not None and hasattr(_agent, 'model') and not _run_failed:
                _cfg_model = _resolve_gateway_model()
                # Normalize _cfg_model the same way AIAgent.__init__ does, so a
                # vendor-prefixed config value (e.g. "deepseek/deepseek-v4-pro")
                # matches the agent's stripped model ("deepseek-v4-pro") on
                # native providers. Without this, _agent.model != _cfg_model is
                # always true for vendor-prefixed config and the cached agent is
                # evicted on every successful turn — destroying prompt caching.
                # Aggregators (openrouter, etc.) keep the vendor/model slug, so
                # they're left untouched.
                try:
                    from hermes_cli.model_normalize import (
                        _AGGREGATOR_PROVIDERS,
                        normalize_model_for_provider,
                    )
                    _agent_provider = getattr(_agent, 'provider', '') or ''
                    if _agent_provider and _agent_provider not in _AGGREGATOR_PROVIDERS:
                        _cfg_model = normalize_model_for_provider(_cfg_model, _agent_provider)
                except Exception:
                    pass
                if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
                    # Fallback activated on a successful run — evict cached
                    # agent so the next message retries the primary model.
                    self._evict_cached_agent(session_key)

            # Check if we were interrupted OR have a queued message (/queue).
            result = result_holder[0]
            adapter = self._adapter_for_source(source)

            # Finalize the streaming-TTS consumer (#60671).
            #
            # finish() is called from the outer event-loop thread (not the
            # executor worker) so early returns from run_sync are also
            # finalised.  wait_complete() drains queued audio; on timeout
            # the consumer is aborted unconditionally — if audio was
            # audible, suppression is preserved so the gateway does not
            # replay from the beginning; if no audio was audible, the
            # whole-file fallback path is permitted.
            _stts = streaming_tts_consumer_holder[0]
            if _stts is not None:
                _stts.finish()
                try:
                    await _stts.wait_complete(timeout=10.0)
                except Exception as _stts_done_err:
                    logger.debug("streaming TTS wait_complete error: %s", _stts_done_err)
                if not _stts.done:
                    # Timeout before or after audible audio: abort to free
                    # the consumer task.  Audible streams retain suppression;
                    # silent streams remain eligible for whole-file fallback.
                    _stts.abort("streaming TTS finalisation timeout")
                    await _stts.wait_complete(timeout=2.0)
                if _stts.suppress_whole_file and adapter is not None:
                    _mark_turn = getattr(adapter, "_mark_streaming_tts_completed_turn", None)
                    if callable(_mark_turn):
                        _mark_turn(session_key, run_generation)

            # Get pending message from adapter.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending_event = None
            pending = None
            if result and adapter and session_key:
                pending_event = _dequeue_pending_event(adapter, session_key)
                # /queue overflow: after consuming the adapter's "next-up"
                # slot, promote the next queued event into it so the
                # recursive run's drain will see it.  This keeps the slot
                # occupied for the full FIFO chain, which (a) preserves
                # order, and (b) causes any mid-chain /queue to correctly
                # route to overflow rather than jumping the queue.
                pending_event = self._promote_queued_event(session_key, adapter, pending_event)
                if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                    interrupt_message = result.get("interrupt_message")
                    if _is_control_interrupt_message(interrupt_message):
                        logger.info(
                            "Ignoring control interrupt message for session %s: %s",
                            session_key or "?",
                            interrupt_message,
                        )
                    else:
                        pending = interrupt_message
                elif pending_event:
                    # Transcribe audio media on the dequeued event BEFORE it is
                    # handed back as the next user turn, so queued/interrupting
                    # voice messages drain with the real transcript instead of
                    # a file-path placeholder. When configured, echo each
                    # transcript back to the user in the same 🎙️ format as
                    # fresh voice messages.
                    _pending_text = pending_event.text or ""
                    _media_urls = getattr(pending_event, "media_urls", None) or []
                    if self._pending_event_audio_paths(pending_event):
                        pending, _ = await self._transcribe_and_echo_pending_voice(
                            pending_event,
                            adapter,
                            source,
                            _pending_text,
                            log_context="Voice-drain",
                            metadata={"thread_id": source.thread_id} if source.thread_id else None,
                        )
                        if not pending:
                            pending = _build_media_placeholder(pending_event)
                    else:
                        pending = _pending_text or _build_media_placeholder(pending_event)
                    if pending:
                        logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

            # Leftover /steer: if a steer arrived after the last tool batch
            # (e.g. during the final API call), the agent couldn't inject it
            # and returned it in result["pending_steer"]. Deliver it as the
            # next user turn so it isn't silently dropped.
            if result and not pending and not pending_event:
                _leftover_steer = result.get("pending_steer")
                if _leftover_steer:
                    pending = _leftover_steer
                    logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

            # Safety net: if the pending text is a slash command (e.g. "/stop",
            # "/new"), discard it — commands should never be passed to the agent
            # as user input.  The primary fix is in base.py (commands bypass the
            # active-session guard), but this catches edge cases where command
            # text leaks through the interrupt_message fallback.
            if pending and pending.strip().startswith("/"):
                _pending_parts = pending.strip().split(None, 1)
                _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
                if _pending_cmd_word:
                    try:
                        from hermes_cli.commands import resolve_command as _rc_pending
                        if _rc_pending(_pending_cmd_word):
                            logger.info(
                                "Discarding command '/%s' from pending queue — "
                                "commands must not be passed as agent input",
                                _pending_cmd_word,
                            )
                            pending_event = None
                            pending = None
                    except Exception:
                        pass

            if self._draining and (pending_event or pending):
                logger.info(
                    "Discarding pending follow-up for session %s during gateway %s",
                    session_key or "?",
                    self._status_action_label(),
                )
                pending_event = None
                pending = None

            if pending_event or pending:
                logger.debug("Processing pending message: '%s...'", pending[:40])

                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                    adapter._active_sessions[session_key].clear()

                # Cap recursion depth to prevent resource exhaustion when the
                # user sends multiple messages while the agent keeps failing. (#816)
                if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                    logger.warning(
                        "Interrupt recursion depth %d reached for session %s — "
                        "queueing message instead of recursing.",
                        _interrupt_depth, session_key,
                    )
                    adapter = self._adapter_for_source(source)
                    if adapter and pending_event:
                        merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
                    elif adapter and hasattr(adapter, 'queue_message'):
                        adapter.queue_message(session_key, pending)
                    return result_holder[0] or {"final_response": response, "messages": history}

                was_interrupted = result.get("interrupted")
                if not was_interrupted:
                    # Queued message after normal completion — deliver the first
                    # response before processing the queued follow-up.
                    # Skip if streaming already delivered it.
                    _sc = stream_consumer_holder[0]
                    if _sc and stream_task:
                        try:
                            await asyncio.wait_for(stream_task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            stream_task.cancel()
                            try:
                                await stream_task
                            except asyncio.CancelledError:
                                pass
                        except Exception as e:
                            logger.debug("Stream consumer wait before queued message failed: %s", e)
                    # The queued branch needs raw ``result`` for interruption,
                    # history, and recursion state, but delivery must use the
                    # finalized task result. The latter contains empty/failure
                    # normalization and any final response processing applied by
                    # _run_agent_task; sending the raw copy bypasses those steps.
                    _delivery_result = response if isinstance(response, dict) else (result or {})
                    _previewed = bool(_delivery_result.get("response_previewed"))
                    first_response = _delivery_result.get("final_response", "")
                    _already_streamed = _stream_confirmed_final_delivery(
                        _sc,
                        first_response,
                        previewed=_previewed,
                    )
                    # Apply the same predicate as the normal completed-turn path.
                    # This direct queued-send branch predates intentional-silence
                    # filtering, so without this check it leaks the literal marker.
                    try:
                        from gateway.response_filters import is_intentional_silence_agent_result
                        _intentional_silence = is_intentional_silence_agent_result(
                            _delivery_result, first_response,
                        )
                    except Exception:
                        _intentional_silence = False
                    if _intentional_silence:
                        logger.info(
                            "Queued follow-up for session %s: suppressing intentional silence marker before continuing.",
                            session_key or "?",
                        )
                    elif first_response:
                        try:
                            if _already_streamed:
                                logger.info(
                                    "Queued follow-up for session %s: final text delivery confirmed; delivering explicit media before continuing.",
                                    session_key or "?",
                                )
                            else:
                                logger.info(
                                    "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                                    session_key or "?",
                                )
                            await self._deliver_queued_first_response(
                                first_response,
                                source=source,
                                adapter=adapter,
                                metadata=_status_thread_metadata,
                                event_message_id=event_message_id,
                                text_already_delivered=_already_streamed,
                                deliver_media=not _delivery_result.get("failed"),
                                stream_consumer=_sc,
                            )
                        except Exception as e:
                            logger.warning("Failed to send first response before queued message: %s", e)
                    # Release deferred bg-review notifications now that the
                    # first response has been delivered.  Pop from the
                    # adapter's callback dict (prevents double-fire in
                    # base.py's finally block) and call it.
                    if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
                        _bg_cb = adapter.pop_post_delivery_callback(
                            session_key,
                            generation=run_generation,
                        )
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                    elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
                        _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                # else: interrupted — discard the interrupted response ("Operation
                # interrupted." is just noise; the user already knows they sent a
                # new message).

                updated_history = result.get("messages", history)
                next_source = source
                next_message = pending
                next_message_id = None
                next_channel_prompt = None
                next_session_key = session_key
                # #60671 — carry the pending event's message_type into the
                # recursive call so queued voice turns can stream TTS and
                # re-mark the generation for the final delivered turn.
                next_message_type = None
                if pending_event is not None:
                    next_source = getattr(pending_event, "source", None) or source
                    if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                        logger.info(
                            "Discarding stale goal continuation for session %s — goal is no longer active",
                            session_key or "?",
                        )
                        return result
                    # Resolve the follow-up's session key BEFORE preparing the
                    # inbound text: _prepare_inbound_message_text buffers native
                    # image paths under the key it is given, and the recursive
                    # _run_agent below consumes them under next_session_key.
                    # The write and consume keys must match or the images drop.
                    try:
                        next_session_key = self._session_key_for_source(next_source)
                    except Exception:
                        logger.debug(
                            "Queued follow-up session-key resolution failed; reusing %s",
                            session_key or "?",
                            exc_info=True,
                        )
                    next_message = await self._prepare_profile_scoped_inbound_message_text(
                        event=pending_event,
                        source=next_source,
                        history=updated_history,
                        session_key=next_session_key,
                    )
                    if next_message is None:
                        return result
                    next_message_id = self._reply_anchor_for_event(pending_event)
                    next_channel_prompt = getattr(pending_event, "channel_prompt", None)
                    next_message_type = getattr(pending_event, "message_type", None)

                # Clear the completed streaming marker from the prior logical
                # turn so the recursive turn's streaming TTS is not suppressed
                # by the prior turn's completion (#60671).
                _clear_adapter = self._adapter_for_source(source)
                if _clear_adapter is not None and session_key and run_generation is not None:
                    _completed_turns = getattr(_clear_adapter, "_streaming_tts_completed_turns", None)
                    if _completed_turns is not None:
                        _prior_key = getattr(_clear_adapter, "_streaming_tts_turn_key", None)
                        if callable(_prior_key):
                            _pk = _prior_key(session_key, run_generation)
                            if _pk:
                                _completed_turns.discard(_pk)

                # Restart typing indicator so the user sees activity while
                # the follow-up turn runs.  The outer _process_message_background
                # typing task is still alive but may be stale.
                _followup_adapter = self._adapter_for_source(source)
                if _followup_adapter:
                    try:
                        await _followup_adapter.send_typing(
                            source.chat_id,
                            metadata=_status_thread_metadata,
                        )
                    except Exception:
                        pass

                # Re-baseline the cached agent's message_count snapshot before
                # recursing into the in-band queued (/queue) follow-up turn.
                # The first turn has completed and flushed its own user +
                # assistant rows to the SessionDB, so the cross-process
                # coherence guard (#45966) — which this recursive _run_agent
                # call re-enters — would otherwise see the grown on-disk count
                # against the stale build-time snapshot and rebuild the agent
                # on THIS process's OWN writes, destroying the prompt-cache
                # prefix #46237 was merged to preserve.  The existing
                # re-baseline in _handle_message_with_agent only runs after the
                # whole _run_agent chain unwinds — too late for the in-band
                # follow-up.  Use the same (session_key, session_id) the
                # recursive call runs under so the snapshot matches exactly
                # what the follow-up's guard will consult.  Fail-safe in helper.
                await self._refresh_agent_cache_message_count(session_key, session_id)

                followup_result = await self._run_agent(
                    message=next_message,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=next_source,
                    session_id=session_id,
                    session_key=next_session_key,
                    run_generation=run_generation,
                    _interrupt_depth=_interrupt_depth + 1,
                    event_message_id=next_message_id,
                    channel_prompt=next_channel_prompt,
                    message_type=next_message_type,
                )
                return _preserve_queued_followup_history_offset(result, followup_result)
        finally:
            # Stop progress sender, interrupt monitor, and notification task
            if progress_task:
                progress_task.cancel()
            if log_task:
                log_task.cancel()
            interrupt_monitor.cancel()
            _notify_task.cancel()

            # Wait for stream consumer to finish its final edit
            if stream_task:
                # If the agent never created a stream consumer (e.g. non-
                # streaming code path, or a test stub returning synchronously)
                # there is nothing to flush — cancel immediately instead of
                # waiting out the 5s timeout on a task that's just polling for
                # a consumer that will never arrive.  This was a 5-second
                # cost per non-streaming test run.
                _has_stream_consumer = (
                    stream_consumer_holder
                    and stream_consumer_holder[0] is not None
                )
                if not _has_stream_consumer:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                else:
                    try:
                        await asyncio.wait_for(stream_task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass

            # Unconditional abort + bounded wait for the streaming-TTS
            # consumer (#60671 hardening).  Covers cancellation / exception
            # paths where the normal finalisation block was skipped.
            _stts_finally = streaming_tts_consumer_holder[0]
            if _stts_finally is not None and not _stts_finally.done:
                _stts_finally.abort("cleanup")
                try:
                    await _stts_finally.wait_complete(timeout=2.0)
                except Exception:
                    pass

            # Clean up tracking
            tracking_task.cancel()
            if session_key:
                # Only release the slot if this run's generation still owns
                # it.  A /stop or /new that bumped the generation while we
                # were unwinding has already installed its own state; this
                # guard prevents an old run from clobbering it on the way
                # out.
                self._release_running_agent_state(
                    session_key, run_generation=run_generation
                )
            if self._draining:
                self._update_runtime_status("draining")

            # Wait for cancelled tasks
            for task in [progress_task, log_task, interrupt_monitor, tracking_task, _notify_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        # A background task that died of a non-cancellation
                        # error (transport drop in a progress/card publish)
                        # must not abort the cleanup path — everything after
                        # this loop (final-delivery bookkeeping) still runs
                        # (review B7).
                        logger.debug(
                            "background turn task failed during cleanup",
                            exc_info=True,
                        )

        # If streaming already delivered the response, mark it so the
        # caller's send() is skipped (avoiding duplicate messages).
        # BUT: never suppress delivery when the agent failed — the error
        # message is new content the user hasn't seen, and it must reach
        # them even if streaming had sent earlier partial output.
        #
        # Also never suppress when the final response is "(empty)" — this
        # means the model failed to produce content after tool calls (common
        # with mimo-v2-pro, GLM-5, etc.).  The stream consumer may have
        # sent intermediate text ("Let me search for that…") alongside the
        # tool call, setting already_sent=True, but that text is NOT the
        # final answer.  Suppressing delivery here leaves the user staring
        # at silence.  (#10xxx — "agent stops after web search")
        _sc = stream_consumer_holder[0]
        if isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            # response_previewed means the interim_assistant_callback already
            # saw the final text, but only suppress the normal send if that
            # exact final text was delivered. Unrelated commentary/progress
            # must not be mistaken for the final response (#14238).
            _previewed = bool(response.get("response_previewed"))
            _content_delivered = bool(
                _sc and getattr(_sc, "final_content_delivered", False)
            )
            # #71643: a *successful* finalize edit can still carry only the
            # last preview snapshot — deltas generated between that edit and
            # stream completion never reach any API call, and both suppression
            # flags are set from the call's success rather than its content.
            # Reconcile the consumer's recorded turn-final payload against the
            # completed response: on a demonstrable mismatch (False) neither
            # final_response_sent nor final_content_delivered may suppress the
            # normal final send. False also covers payload-less multi-message
            # split delivery (#78541). None (no record on a non-split legacy
            # path) keeps legacy trust; the failed-finalize family
            # (#51828 / #33793) is unaffected because those paths leave the
            # flags False or record the complete fallback payload.
            _stale_finalized = False
            if _content_delivered and not _is_empty_sentinel:
                _matcher = getattr(_sc, "delivered_final_matches", None)
                if callable(_matcher):
                    try:
                        _stale_finalized = _matcher(_final) is False
                    except Exception:
                        _stale_finalized = False
                if _stale_finalized:
                    _content_delivered = False
            # Plugin hooks (e.g. transform_llm_output) may have appended content
            # after streaming finished — when the response was transformed, always
            # send the final version so the appended content reaches the client.
            _transformed = bool(response.get("response_transformed"))
            # Only suppress the normal send when the actual final reply reached
            # the user: the stream consumer streamed it (final_response_sent /
            # final_content_delivered), or the interim preview delivered that
            # *exact* final text. Unrelated commentary/progress shown during a
            # compression/session split must not be mistaken for the final
            # response (#14238).
            _streamed = _stream_confirmed_final_delivery(
                _sc,
                _final,
                previewed=_previewed,
            )
            if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
                logger.info(
                    "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                )
                response["already_sent"] = True
            elif not _is_empty_sentinel and not _transformed and _stale_finalized and _sc is not None:
                # Stale finalize (#71643): the streamed message holds only the
                # last preview snapshot. Prefer editing it up to the complete
                # response (same shape as the transformed branch below) so the
                # user gets one corrected message; on edit failure fall through
                # with already_sent unset so the normal final send delivers the
                # complete text.
                #
                # Not valid for a multi-message split delivery: there
                # ``message_id`` is only the LAST chunk, so editing it with the
                # complete response would repeat every sealed head chunk's text
                # inside the tail message. Fall through to the normal final send
                # instead (#78541).
                _sc_msg_id = _sc.message_id
                _sc_adapter = getattr(_sc, "adapter", None)
                if getattr(_sc, "_turn_split_delivery", False):
                    logger.info(
                        "Stale streamed finalize detected for session %s on a multi-message split; skipping the in-place reconciliation edit and delivering the complete response via normal final send (#78541).",
                        session_key or "?",
                    )
                elif _sc_msg_id and _sc_msg_id != "__no_edit__" and _sc_adapter is not None:
                    try:
                        _reconcile_res = await _sc_adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=_final,
                            finalize=True,
                        )
                        if getattr(_reconcile_res, "success", True):
                            response["already_sent"] = True
                            logger.info(
                                "Reconciled stale streamed finalize for session %s: edited message %s with the complete response (#71643).",
                                session_key or "?", _sc_msg_id,
                            )
                        else:
                            logger.warning(
                                "Stale-finalize reconciliation edit failed for session %s (%s); sending complete response via normal final send.",
                                session_key or "?",
                                getattr(_reconcile_res, "error", None),
                            )
                    except Exception as _edit_err:
                        logger.warning(
                            "Stale-finalize reconciliation edit failed for session %s: %s; sending complete response via normal final send.",
                            session_key or "?", _edit_err,
                        )
                else:
                    logger.info(
                        "Stale streamed finalize detected for session %s with no editable message; delivering complete response via normal final send (#71643).",
                        session_key or "?",
                    )
            elif not _is_empty_sentinel and _transformed and _sc is not None:
                # Plugin hooks transformed the response after streaming — edit the
                # existing streamed message instead of sending a duplicate.
                _sc_msg_id = _sc.message_id
                if _sc_msg_id:
                    try:
                        await _sc.adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=response["final_response"],
                            finalize=True,
                        )
                        response["already_sent"] = True
                        logger.info(
                            "Edited streamed message %s for session %s to include plugin-transformed content.",
                            _sc_msg_id, session_key or "?",
                        )
                    except Exception as _edit_err:
                        logger.warning(
                            "Failed to edit streamed message for session %s: %s",
                            session_key or "?", _edit_err,
                        )

        # Schedule deletion of tracked temporary progress bubbles after the
        # final response lands. Failed runs skip this so bubbles remain as
        # breadcrumbs for the user to see what work happened. Only fires on
        # adapters that support ``delete_message`` (see init above); failures
        # are swallowed — deletion is best-effort.
        if (
            _cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            _ids_snapshot = list(_cleanup_msg_ids)
            _chat_id_snapshot = source.chat_id
            _adapter_snapshot = _cleanup_adapter
            _loop_snapshot = asyncio.get_running_loop()

            def _cleanup_temp_bubbles() -> None:
                async def _delete_all() -> None:
                    for _mid in _ids_snapshot:
                        try:
                            await _adapter_snapshot.delete_message(
                                _chat_id_snapshot, _mid
                            )
                        except Exception:
                            pass
                try:
                    safe_schedule_threadsafe(
                        _delete_all(), _loop_snapshot,
                        logger=logger,
                        log_message="Temp bubble cleanup scheduling error",
                    )
                except Exception:
                    pass

            try:
                _cleanup_adapter.register_post_delivery_callback(
                    session_key,
                    _cleanup_temp_bubbles,
                    generation=run_generation,
                )
            except Exception as _rpe:
                logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

        return response
    @dataclasses.dataclass
    class _RunAgentDisplay:
        """Per-turn display / progress settings resolved by ``_run_agent_display_settings``."""
        user_config: Any = None
        platform_key: Any = None
        enabled_toolsets: Any = None
        disabled_toolsets: Any = None
        resolve_display_setting: Any = None
        progress_mode: Any = None
        progress_grouping: Any = None
        _display_surface_mode: Any = None
        tool_progress_enabled: Any = None
        _live_status_mode: Any = None
        _live_status_adapter: Any = None
        log_mode_enabled: Any = None
        log_queue: Any = None
        interim_assistant_messages_enabled: Any = None
        _thinking_enabled: Any = None
        _native_slack_task_cards: Any = None
        needs_progress_queue: Any = None
        _generic_status_phrase: Any = None

    @dataclasses.dataclass
    class _RunAgentWorker:
        """Executor future + inactivity-watchdog handles for one ``_run_agent_inner`` turn."""
        executor_task: Any = None
        agent_timeout: Optional[float] = None
        agent_warning: Optional[float] = None
        task_id: str = ""
        process_baseline: Any = None
        worker_done: Any = None
        timeout_fired: Any = None
        cleanup_lock: Any = None
        is_current: Any = None


def _run_planned_stop_watcher(
    stop_event: threading.Event, runner, loop: asyncio.AbstractEventLoop, shutdown_handler, *,
    poll_interval: float = 0.5) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown (Windows lacks
    ``add_signal_handler``, so ``hermes gateway stop`` would never drain). Runs everywhere; on POSIX
    the signal handler consumes the marker first and ``_running``/``_draining`` guard re-triggers.

    On Windows, ``asyncio.add_signal_handler`` raises NotImplementedError for SIGTERM/SIGINT, so the
    standard signal-driven shutdown path never runs when ``hermes gateway stop`` signals the gateway. The
    consequence is that the drain loop is skipped — in-flight agent sessions are killed mid-turn and
    ``resume_pending`` is never set, so the next gateway boot has no idea those sessions need to be
    auto-resumed (issue #33778, v0.13.0 session-resume feature broken on native Windows).
    """
    from gateway.status import (
        _get_planned_stop_marker_path, planned_stop_marker_targets_self)
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)):
                # A marker may target a PREVIOUS instance that exited before stop() cleaned up;
                # firing on it means an "UNKNOWN" exit and a watchdog crash-loop; probe unlinks stale.
                # A marker existing is NOT sufficient — it may have been written for a PREVIOUS gateway
                # instance (different PID) and left behind because that process exited before the CLI's
                # stop() could clean it up. Firing the handler on a stale/foreign marker drives the gateway
                # into shutdown, then consume_planned_stop_marker_for_self() correctly reports a PID
                # mismatch — but by then we're already stopping, so it's logged as an unexpected "UNKNOWN"
                # exit and the watchdog crash-loops the gateway (issue #34597, a regression from PR #33798
                # which added this watcher without the PID check). Only fire when the marker actually
                # targets us. The probe is non-destructive on a match (the handler does the authoritative
                # consume on the loop thread) and self-heals by unlinking stale/malformed markers so they
                # cannot wedge a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Same path as a real signal; the handler consumes the marker (validates pid + start_time).
                loop.call_soon_threadsafe(shutdown_handler, None)
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)


def _housekeeping_chore(label: str, fn, *args, **kwargs) -> None:
    """Run one housekeeping chore; failures log at debug (a persistent failure such as a broken
    import after a partial update would otherwise warn every tick forever) and never stop the loop."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("%s error: %s", label, exc)


def _housekeeping_channel_directory(adapters, loop) -> None:
    from gateway.channel_directory import build_channel_directory
    if loop is not None:
        # build_channel_directory is async (Slack web calls) and this is a background thread:
        # schedule onto the gateway loop and wait briefly so refresh failures still log.
        fut = safe_schedule_threadsafe(
            build_channel_directory(adapters), loop, logger=logger,
            log_message="Channel directory refresh scheduling error")
        if fut is not None:
            fut.result(timeout=30)


def _housekeeping_media_caches() -> None:
    """Every platform media cache prunes on the same hourly cadence (24h max age)."""
    from gateway.platforms.base import (
        cleanup_audio_cache, cleanup_document_cache, cleanup_image_cache, cleanup_screenshot_cache,
        cleanup_video_cache)
    from tools.tool_result_storage import cleanup_spillover_cache
    from tools.environments.local import cleanup_terminal_temp_cache
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_relay import cleanup_bot_relay_artifacts

    for cache_name, cleanup_fn in (
        ("Image", cleanup_image_cache), ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache), ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache), ("Spillover", cleanup_spillover_cache),
        ("Terminal temp", cleanup_terminal_temp_cache), ("Bot DM", cleanup_bot_dm_cache),
        ("Bot relay", cleanup_bot_relay_artifacts)):
        def _one(name=cache_name, fn=cleanup_fn):
            removed = fn(max_age_hours=24)
            if removed:
                logger.info("%s cache cleanup: removed %d stale file(s)", name, removed)
        _housekeeping_chore(f"{cache_name} cache cleanup", _one)


def _housekeeping_paste_sweep() -> None:
    from hermes_cli.debug import _sweep_expired_pastes
    deleted, remaining = _sweep_expired_pastes()
    if deleted:
        logger.info("Paste sweep: deleted %d expired paste(s), %d pending", deleted, remaining)


def _housekeeping_misfire_catch_up(cron_provider, adapters, loop) -> None:
    """External cron providers only: fire jobs whose time passed with no external fire delivered (dead
    loopback hop). No-op for the built-in ticker; enforces misfire_grace_minutes; CAS claim de-dupes."""
    from cron.scheduler_provider import fire_overdue_jobs
    caught_up = fire_overdue_jobs(cron_provider, adapters=adapters, loop=loop)
    if caught_up:
        logger.info("Misfire catch-up: fired %d overdue job(s)", caught_up)


def _housekeeping_curator() -> None:
    """maybe_run_curator() is gated by config.interval_hours (7 days default); this is the poll."""
    from agent.curator import maybe_run_curator
    maybe_run_curator(idle_for_seconds=float("inf"), on_summary=lambda msg: logger.info("curator: %s", msg))


def _housekeeping_skill_sync() -> None:
    """Inert unless the access gate is open and a sync base URL is configured."""
    from tools.skills_sync_client import maybe_pull_skills
    maybe_pull_skills()


def _housekeeping_org_skill_sync() -> None:
    """Gated on real org membership (the token must carry an org role): solo accounts never reach the network."""
    from tools.skills_sync_client_org import maybe_pull_org_skills
    maybe_pull_org_skills()


def _housekeeping_auto_archive() -> None:
    """Stale-session auto-archive on a live timer (the startup hook fires once); maybe_auto_archive()
    is gated by sessions.min_interval_hours. Opens its own SessionDB — SQLite connections are thread-bound."""
    from hermes_cli.config import load_config as _load_full_config
    from hermes_state_registry import acquire, release_or_close
    _sess_cfg = (_load_full_config().get("sessions") or {})
    if _sess_cfg.get("auto_archive", False):
        _adb = acquire()
        try:
            _adb.maybe_auto_archive(
                idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
        finally:
            release_or_close(_adb)


def _housekeeping_deferred_fts_retry() -> None:
    """A SessionDB opened while another process held the rebuild lock fails closed onto the LIKE fallback
    and the gateway stays up for days. Non-blocking, rate-limited inside SessionDB; no-op when not stale."""
    # Retry here, on the existing tick, against the shared instances this process already holds:
    # non-blocking admission, no new thread, rate-limited inside SessionDB. No-op when nothing is stale (one
    # attribute read per instance). See #100108.
    from hermes_state_registry import borrow_live_shared_session_dbs
    with borrow_live_shared_session_dbs() as _session_dbs:
        for _sdb in _session_dbs:
            _retry = getattr(_sdb, "retry_deferred_fts_recovery", None)
            if callable(_retry) and _retry():
                logger.info(
                    "Deferred state.db FTS rebuild completed in-process for %s; full-text search restored.",
                    getattr(_sdb, "db_path", "state.db"))


def _housekeeping_memory_trim() -> None:
    """Messaging-gateway counterpart to the TUI idle reaper; config-gated and rate-limited inside."""
    from hermes_cli.mem_trim import trim_memory
    trim_memory(reason="messaging gateway housekeeping")


def _drain_restart_safe_cron_deliveries(adapters, loop, runner=None) -> None:
    """Drain each profile's worker queue through its matching live adapters. A credential-less satellite
    profile (empty adapter map) drains through the primary's adapters routed by its own profile routes."""
    from cron import scheduler as cron_scheduler
    from cron import scheduler_preflight as sched_preflight

    if runner is None:
        if adapters is not None:
            cron_scheduler.drain_delivery_queue(adapters, loop)
        return
    for profile_name, profile_home in _handoff_watch_scopes(runner):
        if profile_name is None:
            profile_adapters = adapters
        else:
            profile_adapters = getattr(runner, "_profile_adapters", {}).get(profile_name)
        if profile_adapters is None:
            continue
        with _profile_runtime_scope(profile_home or get_hermes_home()):
            if profile_name is not None and not profile_adapters and adapters:
                routes = sched_preflight._primary_profile_routes_for_current_home()
                if routes:
                    profile_adapters = sched_preflight.SharedRouteAdapters(adapters, routes)
            cron_scheduler.drain_delivery_queue(profile_adapters, loop)


def _start_gateway_housekeeping(
    stop_event: threading.Event, adapters=None, loop=None, interval: int = 60, cron_provider=None, runner=None,
):
    """Background thread for gateway-only periodic chores (NOT cron). Separate from the cron trigger
    so chores run under any ``CronScheduler`` provider (external scale-to-zero has no 60s loop).
    Cadences are ticks of ``interval``; inner gates own the real cadence."""
    chores: list[tuple[int, str, Any]] = []
    if adapters is not None or runner is not None:
        # Restart-safe cron workers run outside the gateway cgroup and queue their final send for
        # whichever gateway is live; drained here (not the scheduler tick) so external providers get it too.
        chores.append((1, "Cron durable delivery queue drain",
                       lambda: _drain_restart_safe_cron_deliveries(adapters, loop, runner)))
    chores += [
        (5, "Channel directory refresh", lambda: adapters and _housekeeping_channel_directory(adapters, loop)),
        (60, "Media cache cleanup", _housekeeping_media_caches),
        (60, "Paste sweep", _housekeeping_paste_sweep)]
    if cron_provider is not None:
        chores.append((5, "Misfire catch-up sweep", lambda: _housekeeping_misfire_catch_up(cron_provider, adapters, loop)))
    chores += [
        (60, "Curator tick", _housekeeping_curator),
        (60, "Sync pull tick", _housekeeping_skill_sync),
        (60, "Org sync pull tick", _housekeeping_org_skill_sync),
        (60, "Auto-archive tick", _housekeeping_auto_archive),
        (1, "Deferred FTS retry tick", _housekeeping_deferred_fts_retry),
        (1, "gateway housekeeping memory trim", _housekeeping_memory_trim)]

    logger.info("Gateway housekeeping started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        tick_count += 1
        for every, label, fn in chores:
            if tick_count % every == 0:
                _housekeeping_chore(label, fn)
        stop_event.wait(timeout=interval)
    logger.info("Gateway housekeeping stopped")


def _start_cron_ticker(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """DEPRECATED shim — runs ONLY the built-in in-process cron tick loop; the trigger now lives behind
    the ``CronScheduler`` provider and housekeeping in ``_start_gateway_housekeeping``."""
    from cron.scheduler_provider import InProcessCronScheduler
    InProcessCronScheduler().start(stop_event, adapters=adapters, loop=loop, interval=interval)


def _stop_cron_provider(provider) -> None:
    """Stop a cron provider without letting it choose the gateway exit code."""
    try:
        provider.stop()
    except SystemExit as exc:
        logger.warning(
            "Cron provider stop() attempted to exit the gateway with code %s; ignoring", exc.code)
    except Exception as exc:
        logger.debug("Cron provider stop() error: %s", exc)


# Cron thread blocks on future.result(timeout=60) (cron/scheduler.py::_deliver_result) + margin.
_CRON_SHUTDOWN_DRAIN_TIMEOUT = 65.0

# Housekeeping's channel-directory refresh blocks on fut.result(timeout=30); cover that + margin.
_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT = 35.0


async def _await_thread_exit(
    thread: Optional[threading.Thread], timeout: float, poll: float = 0.1) -> bool:
    """Wait for a daemon thread to exit WITHOUT blocking the event loop; True if it exited in time.
    A synchronous ``join()`` freezes the loop — fatal for the cron ticker, whose in-flight delivery is a
    coroutine on *this* loop: it could never run, so the join timed out and the message dropped.

    See #58818.
    """
    if thread is None:
        return True
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
    while thread.is_alive() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(poll)
    return not thread.is_alive()


async def _shutdown_mcp_servers_nonblocking(timeout: float = 5.0) -> bool:
    """Close MCP servers off-loop with a bounded wait; True when done within ``timeout``.
    ``shutdown_mcp_servers()`` can block ~15s; on the loop thread short-grace supervisors (s6 3s)
    SIGKILL us before ``mark_exited()`` runs, so every later boot reports a phantom unclean death.
    On timeout shutdown proceeds and the daemon thread is left to finish or die.

    See #82874.
    """
    def _do() -> None:
        try:
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers
            shutdown_mcp_servers()
        except Exception:
            logger.debug("MCP shutdown raised", exc_info=True)

    thread = threading.Thread(target=_do, name="mcp-shutdown", daemon=True)
    thread.start()
    done = await _await_thread_exit(thread, timeout=timeout)
    if not done:
        logger.warning(
            "MCP shutdown did not finish within %.1fs; continuing gateway "
            "teardown (background thread will be reaped at process exit)", timeout)
    return done


def _shutdown_gateway_health_export(runner: Any) -> None:
    """Idempotently drain and detach Gateway Health OTLP export."""
    runtime = getattr(runner, "_gateway_health_export_runtime", None)
    if runtime is None:
        return
    runner._gateway_health_export_runtime = None
    try:
        runtime.shutdown()
    except Exception:
        logger.debug("gateway health OTLP export shutdown failed", exc_info=True)


def _gateway_stderr_formatter() -> logging.Formatter:
    """Return the redacting formatter used by the gateway stderr stream."""
    from agent.redact import RedactingFormatter
    return RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


# ownership guard inserted below (PR #93084)
def _replace_target_belongs_to_other_profile(existing_pid: int) -> bool:
    """Return True when ``--replace`` must refuse to signal ``existing_pid``.
    A poisoned/stale PID record can point at another profile's LIVE gateway (cross-profile SIGTERM
    restart loop). Ownership is decided by the persisted identity record ALONE, bound to the live target
    by exact PID + start-time; live argv can never PROVE ownership (no HERMES_HOME), it is only a
    consistency check. Missing, legacy, conflicting or unprovable identity → refuse (fail closed)."""
    # On Windows there is no systemd/launchd service query at all (_get_service_pids() returns an empty
    # set), so a gateway supervised by a Scheduled Task / Startup VBS looks like an unsupervised orphan to
    # the process scan (#86098). The same holds on every platform for a healthy gateway launched standalone
    # (no service registration) whose PID the runtime record can see (#83683). Exempt the recorded healthy
    # gateway PID and its parent chain: a recorded, liveness-verified gateway is by definition not an orphan
    # "the pidfile/runtime record can't see", and the Scheduled-Task bootstrap's argv (``gateway run``)
    # matches the gateway scan — killing that bootstrap takes the detached gateway it spawned down with it.
    # Exclusion evidence comes from the RAW registration record, not the liveness-validated probe.
    # ``get_running_pid`` (any flags) returns None whenever a record fails validation — start-time mismatch
    # after PID-reuse checks, argv drift, lock hiccups — which is exactly when a healthy standalone gateway
    # (no service supervisor — e.g. `hermes gateway run` on Windows) is at risk: its PID never joins the
    # exclusion set and the sweep hard-kills it. On Windows SIGTERM is TerminateProcess, so the gateway's
    # planned-stop watcher never gets a chance to drain. Reading the raw pidfile + lock records (no
    # validation, no unlink side effects) is strictly safer for a KILL exclusion list: a stale recorded PID
    # at worst spares one process this sweep, while a validation false-negative would kill a live gateway.
    # The validated probe is still consulted for the runtime-status fallback PID it can surface when no
    # pidfile exists.
    try:
        from gateway.status import (
            _get_pid_path, _get_process_hermes_home, _get_process_start_time, _pid_from_record,
            _read_pid_record, _record_looks_like_gateway, _read_process_cmdline, _same_hermes_home)
        our_home = _get_process_hermes_home()

        def refuse(msg: str, *args, level=logging.WARNING) -> bool:
            logger.log(level, "Refusing --replace: " + msg, *args)
            return True

        # Bound claim: the record must name THIS pid with THIS live start time, else it proves nothing.
        record = _read_pid_record(_get_pid_path())
        if not isinstance(record, dict) or not _record_looks_like_gateway(record):
            return refuse("no valid gateway pid record to prove ownership of PID %s.", existing_pid)
        record_pid = _pid_from_record(record)
        if record_pid != existing_pid:
            return refuse("pid record names %s, not target %s.", record_pid, existing_pid)
        recorded_start = record.get("start_time")
        if not isinstance(recorded_start, int) or isinstance(recorded_start, bool):
            return True
        if _get_process_start_time(existing_pid) != recorded_start:
            return refuse("pid record start-time does not match the live process %s (stale/PID-reuse record).",
                          existing_pid)
        recorded_home = record.get("hermes_home")
        if not isinstance(recorded_home, str) or not recorded_home.strip():
            return refuse("pid record predates hermes_home stampings; ownership of PID %s unprovable.",
                          existing_pid)
        if not _same_hermes_home(recorded_home, our_home):
            return refuse("pid record belongs to a different HERMES_HOME (%s, ours %s). Remove the stale PID "
                          "record or stop the owning profile explicitly.", recorded_home, our_home,
                          level=logging.ERROR)
        # Argv never proves ownership; an explicit contradicting --profile / HERMES_HOME= still refuses.
        live_cmdline = _best_effort(lambda: _read_process_cmdline(existing_pid))
        if live_cmdline and _looks_like_profile_conflict_from_cmdline(live_cmdline, our_home):
            return refuse("target PID %s command line explicitly advertises a different profile than "
                          "HERMES_HOME %s.", existing_pid, our_home, level=logging.ERROR)
        return False
    except Exception:
        # Destructive action + unknown ownership => fail closed.
        logger.warning("cross-profile --replace ownership probe failed for PID %s; refusing to signal",
                       existing_pid, exc_info=True)
        return True


def _looks_like_profile_conflict_from_cmdline(command: str, our_home) -> bool:
    """Token-exact contradiction check between a target argv and our home (authority is the pid record).
    Substring matching is not identity: ``--profile timothy`` must NOT read as profile ``tim``. Returns
    False whenever the argv does not clearly contradict our home."""
    from gateway.status import _profile_name_for_home
    profile_name = _profile_name_for_home(our_home)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    def _flag_value(flag: str) -> Optional[str]:
        """Value of ``--flag X`` / ``--flag=X`` occurrences, token-exact."""
        values = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == flag and i + 1 < len(tokens):
                values.append(tokens[i + 1])
                i += 2
                continue
            if tok.startswith(flag + "="):
                values.append(tok[len(flag) + 1:])
            i += 1
        return values[-1] if values else None

    def _env_home_value() -> Optional[str]:
        """HERMES_HOME=<path> env-style assignment on the argv, token-exact."""
        prefix = "HERMES_HOME="
        for tok in reversed(tokens):
            if tok.startswith(prefix):
                return tok[len(prefix):]
        return None

    def _norm(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    for flag in ("--profile", "-p"):
        value = _flag_value(flag)
        if value is None:
            continue
        # Named-profile home: a DIFFERENT explicit profile contradicts it (legacy default argv never carried
        # profile flags). Default/root home: ANY explicit named-profile flag contradicts it.
        if profile_name is None or profile_name == "default" or value != profile_name:
            return True
    home_value = _flag_value("--hermes-home") or _env_home_value()
    return bool(home_value is not None and _norm(home_value) != _norm(str(our_home)))


def _clear_takeover_marker_quiet() -> None:
    """Best-effort: the marker is scoped to one target; a stale one would grief an unrelated shutdown."""
    try:
        from gateway.status import clear_takeover_marker
        clear_takeover_marker()
    except Exception:
        pass


async def _wait_for_pid_exit(pid: int, attempts: int, delay: float) -> bool:
    """Poll for process exit without blocking the loop (a blocking sleep freezes signal handlers and
    health checks). ``os.kill(pid, 0)`` on Windows is NOT a no-op — use the handle-based check."""
    from gateway.status import _pid_exists
    for _ in range(attempts):
        if not _pid_exists(pid):
            return True
        await asyncio.sleep(delay)
    return False


async def _start_gateway_replace_existing_instance(existing_pid: int, replace: bool) -> bool:
    """Handle a live gateway PID under this HERMES_HOME: replace it (``--replace``) or refuse.
    Returns False when startup must abort (refused, permission denied, target still alive)."""
    from gateway.status import get_process_start_time, remove_pid_file, terminate_pid
    if not replace:
        hermes_home = str(get_hermes_home())
        logger.error(
            "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
            "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
            existing_pid, hermes_home)
        print(
            f"\n❌ Gateway already running (PID {existing_pid}).\n"
            f"   Use 'hermes gateway restart' to replace it,\n"
            f"   or 'hermes gateway stop' to kill it first.\n"
            f"   Or use 'hermes gateway run --replace' to auto-replace.\n")
        return False

    # Never signal a process not provably ours (a poisoned PID record → cross-profile restart loop).
    if _replace_target_belongs_to_other_profile(existing_pid):
        from gateway.status import _get_process_hermes_home
        logger.error(
            "Refusing --replace: PID %d cannot be proven to belong "
            "to this profile's gateway (HERMES_HOME %s). Remove the "
            "stale PID record or stop the owning profile explicitly.",
            existing_pid, _get_process_hermes_home())
        return False
    existing_start_time = get_process_start_time(existing_pid)
    logger.info("Replacing existing gateway instance (PID %d) with --replace.", existing_pid)
    # Takeover marker: target exits 0 on our SIGTERM (exit 1 → systemd Restart=on-failure flap loop).
    try:
        from gateway.status import write_takeover_marker
        write_takeover_marker(existing_pid)
    except Exception as e:
        logger.debug("Could not write takeover marker: %s", e)
    # Snapshot children BEFORE signalling: reparented orphans are invisible yet hold scoped token locks.
    try:
        from gateway.status import _snapshot_gateway_children
        _old_gateway_children = _snapshot_gateway_children(existing_pid)
    except Exception:
        _old_gateway_children = []
    try:
        terminate_pid(existing_pid, force=False)
    except ProcessLookupError:
        pass  # Already gone
    except (PermissionError, OSError):
        logger.error("Permission denied killing PID %d. Cannot replace.", existing_pid)
        _clear_takeover_marker_quiet()
        return False
    # Up to 10s for SIGTERM, then SIGKILL.
    if not await _wait_for_pid_exit(existing_pid, 20, 0.5):
        logger.warning("Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.", existing_pid)
        old_gateway_exited = False
        try:
            terminate_pid(existing_pid, force=True, expected_start_time=existing_start_time)
        except ProcessLookupError:
            old_gateway_exited = True
        except (PermissionError, OSError):
            pass
        # Confirm SIGKILL took (D-state/zombie) before clearing PID/locks, or two gateways share a token.
        if not old_gateway_exited and not await _wait_for_pid_exit(existing_pid, 20, 0.25):
            logger.error(
                "Old gateway (PID %d) still appears alive after SIGKILL; "
                "aborting replacement to avoid a duplicate gateway.", existing_pid)
            _clear_takeover_marker_quiet()
            return False
    # Reap orphaned children (POSIX; mirrors Windows taskkill /T) so they stop holding scoped token locks.
    try:
        from gateway.status import reap_gateway_children
        reap_gateway_children(_old_gateway_children, parent_pid=existing_pid)
    except Exception:
        logger.debug("Child reap for replaced gateway PID %d failed", existing_pid, exc_info=True)
    remove_pid_file()
    # remove_pid_file() is a no-op when the PID doesn't match; force-unlink covers a crashed old process.
    with suppress(Exception):
        (get_hermes_home() / "gateway.pid").unlink(missing_ok=True)
    # The old process may not have consumed the marker (SIGKILL'd before its handler read it).
    _clear_takeover_marker_quiet()
    # Stopped (Ctrl+Z) processes don't release scoped locks on exit; stale lock files block the new gateway.
    try:
        from gateway.status import release_all_scoped_locks
        _released = release_all_scoped_locks(owner_pid=existing_pid, owner_start_time=existing_start_time)
        if _released:
            logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
    except Exception:
        pass
    return True


def _start_gateway_configure_logging(verbosity: Optional[int]) -> None:
    """Sync bundled skills, set up file logging + startup security audit, and the -v/-q stderr handler."""
    def _sync_skills() -> None:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)

    _best_effort(_sync_skills)

    # Centralized logging (agent.log INFO+, errors.log WARNING+, gateway.log gateway-only); idempotent.
    from hermes_logging import setup_logging, _safe_stderr
    setup_logging(hermes_home=_hermes_home, mode="gateway")

    def _security_audit() -> None:
        # Warn-on-load, never blocks: surfaces root / weak-SSH / unauthenticated-listener exposure.
        from hermes_cli.security_audit_startup import log_startup_security_warnings

        def _raw_cfg():
            from hermes_cli.config import read_raw_config
            return read_raw_config()

        log_startup_security_warnings(hermes_home=_hermes_home, config=_best_effort(_raw_cfg))

    _best_effort(_security_audit, "Startup security audit failed (non-fatal): %s")

    # Optional stderr handler from -v/-q: None (quiet) = none; 0 = WARNING; 1 = INFO; 2+ = DEBUG.
    if verbosity is not None:
        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler(_safe_stderr())
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(_gateway_stderr_formatter())
        root = logging.getLogger()
        root.addHandler(_stderr_handler)
        if _stderr_level < root.level:  # so DEBUG records can reach the handler
            root.setLevel(_stderr_level)


def _start_gateway_make_shutdown_signal_handler(runner, _signal_initiated_shutdown: list):
    """Build the SIGINT/SIGTERM handler; ``_signal_initiated_shutdown[0]`` records an unplanned signal."""
    def shutdown_signal_handler(received_signal=None):
        # Planned --replace takeover (sibling marked this PID): exit 0 so systemd won't revive us.
        def _takeover() -> bool:
            from gateway.status import consume_takeover_marker_for_self
            return consume_takeover_marker_for_self()

        # Planned stop: CLI marks first, else its SIGTERM looks like an external kill. SIGINT = Ctrl+C.
        def _planned_stop() -> bool:
            from gateway.status import consume_planned_stop_marker_for_self
            return consume_planned_stop_marker_for_self()

        # Fast (<10ms) sync snapshot: stdlib + /proc, no subprocesses (`ps aux` here once blocked ~3s).
        def _snapshot():
            from gateway.shutdown_forensics import snapshot_shutdown_context
            return snapshot_shutdown_context(received_signal)

        planned_takeover = bool(_best_effort(_takeover, "Takeover marker check failed: %s"))
        planned_stop = received_signal == signal.SIGINT or (
            not planned_takeover and bool(_best_effort(_planned_stop, "Planned stop marker check failed: %s")))
        _shutdown_ctx = _best_effort(_snapshot, "snapshot_shutdown_context failed: %s")
        sig_name = _shutdown_ctx["signal"] if _shutdown_ctx else None

        if planned_takeover:
            logger.info("Received %s as a planned --replace takeover — exiting cleanly", sig_name or "SIGTERM")
        elif planned_stop:
            logger.info("Received %s as a planned gateway stop — exiting cleanly", sig_name or "SIGTERM/SIGINT")
        else:
            # Mirrored onto the runner so _stop_impl suppresses the gateway_state=stopped persist for
            # unexpected signals; operator stops take the `planned_stop` branch and leave it False (DO persist).
            _signal_initiated_shutdown[0] = runner._signal_initiated_shutdown = True
            logger.info("Received %s — initiating shutdown", sig_name or "SIGTERM/SIGINT")

        if _shutdown_ctx is not None:
            def _log_context() -> None:
                # The most useful line for "gateway keeps dying" tickets.
                from gateway.shutdown_forensics import format_context_for_log
                logger.warning("Shutdown context: %s", format_context_for_log(_shutdown_ctx))

            def _diagnostic() -> None:
                # Heavyweight (ps auxf, pstree, dmesg), detached so it finishes even if our cgroup is torn
                # down; bounded by an internal timeout, never blocks.
                from gateway.shutdown_forensics import spawn_async_diagnostic
                spawn_async_diagnostic(
                    _hermes_home / "logs" / "gateway-shutdown-diag.log", _shutdown_ctx["signal"], timeout_seconds=5.0)

            _best_effort(_log_context, "format_context_for_log failed: %s")
            _best_effort(_diagnostic, "spawn_async_diagnostic failed: %s")
        asyncio.create_task(runner.stop())
    return shutdown_signal_handler


def _start_gateway_claim_pid_file() -> bool:
    """Claim the runtime lock + PID file (O_EXCL winner is the authoritative gateway). False = lost."""
    import atexit
    from gateway.status import (
        acquire_gateway_runtime_lock, get_running_pid, release_gateway_runtime_lock,
        remove_pid_file, write_pid_file)
    _current_pid = get_running_pid()
    if _current_pid is not None and _current_pid != os.getpid():
        logger.error("Another gateway instance (PID %d) started during our startup. "
                     "Exiting to avoid double-running.", _current_pid)
        return False
    if not acquire_gateway_runtime_lock():
        logger.error("Gateway runtime lock is already held by another instance. Exiting.")
        return False
    try:
        write_pid_file()
    except FileExistsError:
        release_gateway_runtime_lock()
        logger.error("PID file race lost to another gateway instance. Exiting.")
        return False
    atexit.register(remove_pid_file)
    atexit.register(release_gateway_runtime_lock)
    return True


async def _start_gateway_start_control_socket(runner):
    """Start the gateway control socket (identify/status/pause-for-update); None when unavailable."""
    import atexit
    _control_server = None
    try:
        # Started immediately after the PID-file claim: winning that O_EXCL race is the moment this process
        # becomes the authoritative gateway for its HERMES_HOME, so from here on "does a socket answer?" is
        # a truthful liveness/identity query for updater and fleet consumers. Strictly non-fatal: a bind
        # failure only means consumers fall back to the process-scan/state-file layer, exactly as before
        # this feature. See #92091.
        from gateway.control_socket import GatewayControlServer
        # pause-for-update: the updater asks us to drain + exit (freeing venv handles) vs. a tree-kill
        # (same path as SIGUSR1). Handler runs on the socket executor thread, so marshal onto the loop.
        # pause-for-update (#92091 step 2): the updater asks this gateway to drain in-flight turns and exit
        # cleanly — releasing every venv file handle — instead of being tree-killed mid-turn. Same drain
        # path as SIGUSR1/service restarts (request_restart(via_service=True)); the updater (or the service
        # manager) relaunches after the code swap.
        _main_loop = asyncio.get_running_loop()

        def _pause_for_update_handler() -> dict:
            try:
                from hermes_cli.gateway import _get_restart_drain_timeout
                _drain = float(_get_restart_drain_timeout())
            except Exception:
                _drain = 30.0
            accepted_box: list[bool] = []
            _done = threading.Event()

            def _request() -> None:
                try:
                    accepted_box.append(runner.request_restart(detached=False, via_service=True))
                finally:
                    _done.set()

            _main_loop.call_soon_threadsafe(_request)
            _done.wait(timeout=5.0)
            accepted = bool(accepted_box and accepted_box[0])
            return {
                "pausing": accepted, "already_stopping": not accepted,
                "pid": os.getpid(), "drain_timeout": _drain}

        _control_server = GatewayControlServer(
            verb_handlers={"pause-for-update": _pause_for_update_handler})
        if not await _control_server.start():
            _control_server = None
        else:
            atexit.register(_control_server.cleanup_files)
    except Exception as _cs_exc:
        logger.debug("Control socket startup failed (non-fatal): %s", _cs_exc)
        _control_server = None
    return _control_server


def _start_gateway_start_cron_and_housekeeping(runner):
    """Start the cron scheduler thread + gateway housekeeping thread; returns
    ``(cron_stop, cron_provider, cron_thread, housekeeping_thread)``."""
    # The event loop is passed so cron delivery can use live adapters (E2EE support).
    from cron.scheduler_provider import (
        InProcessCronScheduler, resolve_cron_scheduler, scheduler_for_profile_mode)
    cron_stop = threading.Event()
    multiplex_cron = bool(getattr(runner.config, "multiplex_profiles", False))
    cron_provider = scheduler_for_profile_mode(
        resolve_cron_scheduler(), multiplex_profiles=multiplex_cron)
    cron_start_kwargs: Dict[str, Any] = {"adapters": runner.adapters, "loop": asyncio.get_running_loop()}

    # Multiplex: tell the ticker which profile homes to tick, else secondary profiles' jobs never run.
    if isinstance(cron_provider, InProcessCronScheduler) and multiplex_cron:
        try:
            profile_homes = _multiplex_profile_homes(runner.config)
            if profile_homes:
                cron_start_kwargs["profile_homes"] = profile_homes
                # Per-profile adapters so each profile's cron output goes via its own bot, not the default's.
                cron_start_kwargs["profile_adapters"] = getattr(runner, "_profile_adapters", None)
                # runner.adapters belongs to "default"; naming it keeps the ticker from routing a secondary's
                # cron through the default bot (even before that profile's adapter connects).
                cron_start_kwargs["default_profile"] = "default"
                logger.info(
                    "Cron scheduler will tick %d profile(s) under multiplex: %s", len(profile_homes),
                    [p[0] if isinstance(p, tuple) else p for p in profile_homes])
        except Exception as exc:
            logger.warning("Could not resolve profile homes for multiplex cron: %s", exc)

    # Only the in-process ticker polls local due jobs, so only it gets the external-drain dispatch gate.
    if isinstance(cron_provider, InProcessCronScheduler):
        cron_start_kwargs["can_dispatch"] = lambda: not (
            runner._draining or runner._external_drain_active)
    cron_thread = threading.Thread(
        target=cron_provider.start, args=(cron_stop,), kwargs=cron_start_kwargs, daemon=True,
        name="cron-scheduler")
    cron_thread.start()

    # External providers fire over loopback HTTP to THIS process's api_server; if it never came up (usually
    # API_SERVER_KEY missing) every fire fails while manual runs work — misread as a job bug. Say it ONCE.
    if not isinstance(cron_provider, InProcessCronScheduler):
        try:
            _has_api_server = Platform.API_SERVER in (runner.adapters or {})
        except Exception:
            _has_api_server = True  # never let the tell break startup
        if not _has_api_server:
            logger.warning(
                "Cron provider '%s' is active but the api_server adapter is "
                "NOT running in this gateway — scheduled fires arrive over "
                "loopback HTTP and will all fail (jobs only run when "
                "triggered manually). Most common cause: API_SERVER_KEY is "
                "missing from this gateway process's environment. Restart "
                "the gateway through its supervisor (`hermes gateway "
                "restart`) so the profile env loads.",
                getattr(cron_provider, "name", "external"))

    # Gateway-only housekeeping runs independently of the cron provider; shares cron_stop for shutdown.
    housekeeping_thread = threading.Thread(
        target=_start_gateway_housekeeping, args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop(),
                "cron_provider": cron_provider, "runner": runner},
        daemon=True, name="gateway-housekeeping")
    housekeeping_thread.start()
    return cron_stop, cron_provider, cron_thread, housekeeping_thread


async def _start_gateway_shutdown_tail(
    runner, _control_server, cron_stop: threading.Event, cron_provider,
    cron_thread: threading.Thread, housekeeping_thread: threading.Thread,
    _planned_stop_watcher_stop: threading.Event, _planned_stop_watcher_thread: threading.Thread,
    _signal_initiated_shutdown: list) -> bool:
    """Post-``wait_for_shutdown`` teardown; returns the process exit verdict (True = exit 0)."""
    # Control socket first: once shutdown begins we are no longer a truthful "serving here" answer and a
    # successor must be able to bind. Early-exit paths rely on the atexit cleanup_files hook instead.
    if _control_server is not None:
        try:
            await _control_server.stop()
        except Exception:
            logger.debug("Control socket stop failed (non-fatal)", exc_info=True)

    def _stop_keepalive() -> None:
        from hermes_cli.nous_auth_keepalive import stop_nous_auth_keepalive
        stop_nous_auth_keepalive()

    _best_effort(_stop_keepalive)
    if _exit_with_failure_verdict(runner):
        return False

    # Never join(): an in-flight cron delivery is a coroutine on THIS loop; a sync join would drop it.
    # Stop cron scheduler + housekeeping cleanly. These MUST be awaited cooperatively, not join()ed. A cron
    # delivery in flight when the gateway restarts is a coroutine scheduled onto THIS event loop
    # (safe_schedule_threadsafe); the ticker thread is blocked on its future.result(). A synchronous
    # cron_thread.join() would block the loop, so that delivery could never run — it timed out and the
    # message was silently dropped (#58818). Awaiting keeps the loop alive so the in-flight delivery
    # finishes before we tear down.
    cron_stop.set()
    _stop_cron_provider(cron_provider)
    if not await _await_thread_exit(cron_thread, timeout=_CRON_SHUTDOWN_DRAIN_TIMEOUT):
        logger.warning("Cron ticker did not exit within %.0fs of shutdown — an in-flight "
                       "delivery may have been dropped.", _CRON_SHUTDOWN_DRAIN_TIMEOUT)
    await _await_thread_exit(housekeeping_thread, timeout=_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT)

    # Stop the planned-stop watcher (daemon=True so this is belt-and-suspenders).
    _planned_stop_watcher_stop.set()
    _planned_stop_watcher_thread.join(timeout=2)

    with suppress(Exception):
        await _shutdown_mcp_servers_nonblocking()

    return _resolve_gateway_exit_verdict(runner, _signal_initiated_shutdown[0])


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """Start the gateway and run until interrupted; False if it failed to start (non-zero exit so
    systemd can auto-restart). ``replace`` kills any existing instance first (avoids restart-loop deadlocks)."""
    # Set here (not at import) so incidental gateway.run imports from CLI code don't poison it.
    os.environ["HERMES_EXEC_ASK"] = "1"

    from hermes_cli.resource_limits import apply_nofile_soft_limit
    apply_nofile_soft_limit()

    # Snapshot the revision while sys.modules matches disk so a later `git pull` is detected safely.
    from gateway.code_skew import record_boot_fingerprint
    record_boot_fingerprint()

    # Duplicate-instance guard scoped to HERMES_HOME; distinct-home multi-profile setups coexist.
    from gateway.status import get_running_pid
    existing_pid = get_running_pid()
    if (existing_pid is not None and existing_pid != os.getpid()
            and not await _start_gateway_replace_existing_instance(existing_pid, replace)):
        return False

    _start_gateway_configure_logging(verbosity)

    runner = GatewayRunner(config)
    # Multiplex: swap the launch-home file handlers for per-profile routers so each profile's records
    # land in its own logs/. Must run after the runner resolved (possibly None) config and setup_logging.
    # See #82936.
    _enable_multiplex_log_routing(runner.config)
    # ``--replace`` is explicit startup authority, not a durable reconnect policy: GatewayRunner scopes
    # it to cold adapter connects and clears it before the background reconnect watcher starts.
    runner._platform_lock_takeover_on_start = bool(replace)

    # Unexpected signals exit non-zero so service managers revive us; planned stops write a marker first.
    _signal_initiated_shutdown = [False]

    shutdown_signal_handler = _start_gateway_make_shutdown_signal_handler(
        runner, _signal_initiated_shutdown)

    def restart_signal_handler():
        runner.request_restart(detached=False, via_service=True)

    loop = asyncio.get_running_loop()

    # Swallow transient network errors from background tasks; one unhandled httpx error would kill us.
    # Issues #31066 / #31110: an unhandled ``telegram.error.TimedOut`` (or peer NetworkError / httpx
    # connection error) in any awaited coroutine would propagate to the loop and kill the gateway process,
    # taking down every profile attached to the same runner. systemd then restarts the service after ~5s but
    # the active conversation turn is lost. The fix is intentionally narrow: only well-known transient
    # network errors are swallowed (and logged with full traceback so the originating call site is still
    # discoverable). Anything else is forwarded to the default handler so real bugs still surface.
    loop.set_exception_handler(_gateway_loop_exception_handler)

    if threading.current_thread() is threading.main_thread():
        # add_signal_handler raises NotImplementedError on Windows; SIGUSR1 is POSIX-only.
        handlers = [(sig, shutdown_signal_handler, (sig,)) for sig in (signal.SIGINT, signal.SIGTERM)]
        if hasattr(signal, "SIGUSR1"):
            handlers.append((signal.SIGUSR1, restart_signal_handler, ()))  # windows-footgun: ok — hasattr-guarded
        for sig, handler, args in handlers:
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, handler, *args)  # windows-footgun: ok — suppress(NotImplementedError)
    else:
        logger.info("Skipping signal handlers (not running in main thread).")

    # Windows has no add_signal_handler, so `hermes gateway stop`'s SIGTERM would never drain; poll the
    # planned-stop marker (written BEFORE the kill) instead. Runs everywhere so masked-SIGTERM drains.
    # Windows fallback: asyncio.add_signal_handler raises NotImplementedError on Windows, so `hermes gateway
    # stop`'s SIGTERM (which Python maps to TerminateProcess on Windows) never invokes
    # shutdown_signal_handler. That means the drain loop never runs, mark_resume_pending never fires, and
    # sessions are silently lost across restarts (issue #33778). The fix is a marker-polling thread: `hermes
    # gateway stop` writes the planned-stop marker BEFORE killing, and this thread notices it and drives the
    # same shutdown path the signal handler would have. Runs on every platform (cheap, defensive) so
    # non-signal-bearing environments (Windows native, sandboxed CI runners that mask SIGTERM) still get a
    # clean drain.
    _planned_stop_watcher_stop = threading.Event()
    _planned_stop_watcher_thread = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(_planned_stop_watcher_stop, runner, loop, shutdown_signal_handler), daemon=True,
        name="planned-stop-watcher")
    _planned_stop_watcher_thread.start()

    # PID file BEFORE adapters: of two concurrent `run --replace`, only the O_EXCL winner opens sockets.
    if not _start_gateway_claim_pid_file():
        return False

    # Right after the PID claim (which makes us authoritative); non-fatal — consumers fall back to scan.
    _control_server = await _start_gateway_start_control_socket(runner)

    def _lifecycle_record_startup() -> None:
        # Report if the previous life died uncleanly (SIGKILL / OOM / VM death), then claim the
        # sentinel for this life. After the PID-file claim so a --replace loser can't clobber it.
        from gateway.lifecycle_ledger import record_startup
        record_startup()

    def _start_keepalive() -> None:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive
        start_nous_auth_keepalive()

    _best_effort(_lifecycle_record_startup, "Lifecycle ledger startup record failed: %s")
    _best_effort(_start_keepalive, "Nous auth keepalive did not start: %s")
    _ensure_windows_gateway_venv_imports()

    # discover_mcp_tools() blocks up to 120s; on the loop thread it would freeze platform heartbeats.
    try:
        # MCP tool discovery — run in an executor so the asyncio event loop stays responsive even when a
        # configured MCP server is slow or unreachable.  discover_mcp_tools() uses a blocking 120s wait
        # internally; calling it from the loop thread would freeze platform heartbeats (Discord shard,
        # Telegram polling) until it returned. See #16856.
        await _discover_gateway_mcp_tools(runner.config)
    except Exception as e:
        logger.debug("MCP tool discovery failed: %s", e)

    try:
        success = await runner.start()
    except BaseException:
        _shutdown_gateway_health_export(runner)
        raise
    if not success:
        _shutdown_gateway_health_export(runner)
        return False

    def _recover_pending() -> None:
        from gateway.shutdown_flush import recover_pending_to_db
        recovered = recover_pending_to_db()
        if recovered:
            logger.info("Recovered %d pending message(s) from shutdown flush", recovered)

    _best_effort(_recover_pending)
    if runner.should_exit_cleanly:
        _shutdown_gateway_health_export(runner)
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        # Explicit exit codes (GATEWAY_FATAL_CONFIG_EXIT_CODE) must propagate so s6 finish maps 78 → 125.
        if runner.exit_code is not None:
            raise SystemExit(runner.exit_code)
        return True
    if not runner._running:
        # Startup aborted by restart/shutdown before running mode; preserve that path without starting cron.
        try:
            await runner.wait_for_shutdown()
            with suppress(Exception):
                await _shutdown_mcp_servers_nonblocking()
            return _resolve_gateway_exit_verdict(runner, _signal_initiated_shutdown[0])
        finally:
            _shutdown_gateway_health_export(runner)

    cron_stop, cron_provider, cron_thread, housekeeping_thread = (
        _start_gateway_start_cron_and_housekeeping(runner))

    # READY only once adapters, cron and housekeeping run; missing systemd state just disables watchdog.
    runner._start_systemd_watchdog()

    await runner.wait_for_shutdown()

    return await _start_gateway_shutdown_tail(
        runner, _control_server, cron_stop, cron_provider, cron_thread, housekeeping_thread,
        _planned_stop_watcher_stop, _planned_stop_watcher_thread, _signal_initiated_shutdown)


def _guard_corrupt_user_config() -> None:
    """Fail closed when the active profile's config.yaml cannot be parsed: nobody can repair it on this
    surface, and defaults would let provider auto-detection adopt ``.env`` credentials the config never
    named. Same policy and escape hatch (``HERMES_IGNORE_USER_CONFIG=1``) as ``hermes_cli/main.py``."""
    from hermes_cli.config import InvalidUserConfigError, require_parseable_user_config

    try:
        require_parseable_user_config()
    except InvalidUserConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main():
    """CLI entry point for the gateway."""
    # Before any config-dependent startup (watchdog, DB opens, provider resolution).
    _guard_corrupt_user_config()

    # Advertise the harness to children (mirrors _advertise_agent_env in hermes_cli/main.py, inlined to
    # avoid its startup side effects). Value must equal registry id ``hermes-agent`` exactly.
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")

    def _register_identity() -> None:
        # Ledger registration + Windows job-object attach so update-time reapers can identify this gateway.
        from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self
        register_self("gateway")
        attach_self_to_kill_on_close_job()

    def _arm_watchdog() -> None:
        # Armed before config load / DB opens so a pre-loop deadlock is respawned by the supervisor instead
        # of wedging as a live-PID zombie. GatewayRunner disarms it.
        from hermes_startup_watchdog import arm_startup_watchdog
        arm_startup_watchdog()

    def _utf8_stdio() -> None:
        # Windows: gateway logs and banner would UnicodeEncodeError on cp1252 consoles. No-op on POSIX.
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()

    for _step in (_register_identity, _arm_watchdog, _utf8_stdio):
        _best_effort(_step)

    import argparse
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            config = GatewayConfig.from_dict(yaml.safe_load(f) or {})

    # start_gateway() completes teardown before returning/raising SystemExit; force-exit after so a
    # wedged non-daemon worker can't block Py_FinalizeEx's join. SystemExit caught so EVERY path exits.
    try:
        # start_gateway() performs the full graceful teardown (adapters disconnected, sessions saved +
        # flushed, SQLite closed, cron/MCP stopped, PID file + runtime lock released) before it returns OR
        # raises SystemExit with an explicit code. Force-exit afterwards so a wedged non-daemon worker
        # thread (e.g. a ThreadPoolExecutor tool/LLM call blocked with no timeout) cannot block interpreter
        # finalization (Py_FinalizeEx joins all non-daemon threads, incl. concurrent.futures' _python_exit)
        # and strand the gateway half-shut down with the supervisor unable to restart it (#53107).
        # SystemExit is caught explicitly: start_gateway raises it on the clean-fatal-config (#51228),
        # planned-restart, and service-restart paths, all of which complete teardown first. Routing those
        # codes through the same os._exit backstop means EVERY exit path is wedge-proof, not just the
        # boolean-return ones.
        success = asyncio.run(start_gateway(config))
        exit_code = 0 if success else 1
    except SystemExit as e:
        # e.code may be None (→ 0), an int, or a str (→ 1, like CPython).
        exit_code = 0 if e.code is None else e.code if isinstance(e.code, int) else 1
    _exit_after_graceful_shutdown(exit_code)


def _exit_after_graceful_shutdown(exit_code: int) -> None:
    """Flush stdio, release the PID file + runtime lock, then hard-exit.
    ``os._exit`` (not ``sys.exit``): SystemExit runs ``Py_FinalizeEx``, which joins every non-daemon
    thread — exactly the hang a wedged worker causes. It bypasses ``atexit``, so PID/lock release and the
    bounded log drain (file handlers sit behind a ``QueueListener`` thread) are done here explicitly.

    Graceful teardown is already complete by the time this runs, so there is nothing left that needs a clean
    interpreter shutdown. See #53107.
    ``os._exit`` bypasses ``atexit`` handlers, so we cannot rely on the ``atexit``-registered
    ``remove_pid_file`` / ``release_gateway_runtime_lock`` (registered in ``start_gateway``) to run. The
    full-shutdown path releases both explicitly in ``_stop_impl``, but the EARLY exit paths —
    clean-fatal-config (#51228) and startup-aborted-before-running — raise ``SystemExit`` right after
    ``runner.start()`` without going through ``_stop_impl``, so on those paths ``atexit`` was the only thing
    releasing them. Now that those paths are routed through this backstop (#53107), release both here
    explicitly. Both calls are idempotent — ``remove_pid_file`` only unlinks a PID file that belongs to this
    process, and ``release_gateway_runtime_lock`` no-ops when the lock is already released — so this is a
    no-op on the normal shutdown path and the actual cleanup on the early-exit paths.
    """
    for stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            stream.flush()
    def _release_locks() -> None:
        # BEFORE the log drain (bounded, but could take its full timeout on a wedged disk); idempotent.
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()

    def _mark_exited() -> None:
        # Single funnel every graceful exit passes through, so the next boot's unclean-death detector
        # fires only for genuine SIGKILL/OOM/VM deaths. Ownership-guarded against an old --replace life.
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason="graceful_shutdown")

    def _drain_logs() -> None:
        # os._exit bypasses the listener's atexit drain. Bounded, no restart — NOT flush_log_queue():
        # a listener wedged on the rotation lock would re-freeze shutdown in an unbounded stop() join.
        from hermes_logging import drain_log_queue
        drain_log_queue(timeout=1.0)

    for _step in (_release_locks, _mark_exited, _drain_logs):
        _best_effort(_step)
    os._exit(exit_code)


if __name__ == "__main__":
    main()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Awaitable  # noqa: F401,E402
from contextvars import Context  # noqa: F401,E402
from typing import Union  # noqa: F401,E402
import faulthandler  # noqa: F401,E402
import functools  # noqa: F401,E402
import inspect  # noqa: F401,E402
from dotenv import load_dotenv  # noqa: F401,E402
import queue  # noqa: F401,E402
from datetime import timedelta  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT': ('gateway.restart', 'DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT'),
    'DEFAULT_HEARTBEAT_INTERVAL_S': ('gateway.shutdown_watchdog', 'DEFAULT_HEARTBEAT_INTERVAL_S'),
    'DEFAULT_LEASE_WAIT': ('gateway.turn_lease', 'DEFAULT_LEASE_WAIT'),
    'DEFAULT_LOOP_WATCHDOG_INTERVAL_S': ('gateway.shutdown_watchdog', 'DEFAULT_LOOP_WATCHDOG_INTERVAL_S'),
    'DEFAULT_LOOP_WATCHDOG_MAX_STRIKES': ('gateway.shutdown_watchdog', 'DEFAULT_LOOP_WATCHDOG_MAX_STRIKES'),
    'DEFAULT_LOOP_WATCHDOG_TIMEOUT_S': ('gateway.shutdown_watchdog', 'DEFAULT_LOOP_WATCHDOG_TIMEOUT_S'),
    'EphemeralReply': ('gateway.platforms.base', 'EphemeralReply'),
    'GATEWAY_FATAL_CONFIG_EXIT_CODE': ('gateway.restart', 'GATEWAY_FATAL_CONFIG_EXIT_CODE'),
    'GATEWAY_SERVICE_RESTART_EXIT_CODE': ('gateway.restart', 'GATEWAY_SERVICE_RESTART_EXIT_CODE'),
    'SessionEntry': ('gateway.session', 'SessionEntry'),
    'TranscriptReadError': ('gateway.session_transcript', 'TranscriptReadError'),
    'TurnContext': ('gateway.turn_context', 'TurnContext'),
    'TurnLeaseTimeoutError': ('gateway.turn_lease', 'TurnLeaseTimeoutError'),
    'TurnRunner': ('gateway.run_turn_runner', 'TurnRunner'),
    'arm_shutdown_watchdog': ('gateway.shutdown_watchdog', 'arm_shutdown_watchdog'),
    'atomic_json_write': ('utils', 'atomic_json_write'),
    'base_url_hostname': ('utils', 'base_url_hostname'),
    'build_auto_tts_output_path': ('gateway.platforms.base', 'build_auto_tts_output_path'),
    'build_channel_continuity_note': ('gateway.session', 'build_channel_continuity_note'),
    'build_session_context': ('gateway.session', 'build_session_context'),
    'build_session_context_prompt': ('gateway.session', 'build_session_context_prompt'),
    'consume_detached_task_result': ('agent.async_utils', 'consume_detached_task_result'),
    'is_global_startup_conflict': ('gateway.restart', 'is_global_startup_conflict'),
    'is_shared_multi_user_session': ('gateway.session', 'is_shared_multi_user_session'),
    'is_truthy_value': ('utils', 'is_truthy_value'),
    'looks_like_telegram_private_chat_id': ('gateway.delivery', 'looks_like_telegram_private_chat_id'),
    'loop_heartbeat_forever': ('gateway.shutdown_watchdog', 'loop_heartbeat_forever'),
    'merge_pending_message_event': ('gateway.platforms.base', 'merge_pending_message_event'),
    'neutralize_untrusted_inline_text': ('gateway.session', 'neutralize_untrusted_inline_text'),
    'parse_cron_drain_timeout': ('gateway.restart', 'parse_cron_drain_timeout'),
    'parse_restart_after_turn_timeout': ('gateway.restart', 'parse_restart_after_turn_timeout'),
    'parse_restart_drain_timeout': ('gateway.restart', 'parse_restart_drain_timeout'),
    'parse_signal_interrupt_grace_timeout': ('gateway.restart', 'parse_signal_interrupt_grace_timeout'),
    'project_compaction_message_for_display': ('agent.compaction_display', 'project_compaction_message_for_display'),
    'repair_explicit_computer_use_media_paths': ('gateway.media_repair', 'repair_explicit_computer_use_media_paths'),
    'resolve_cron_drain_budget': ('gateway.restart', 'resolve_cron_drain_budget'),
    'resolve_delivery_transport': ('gateway.delivery', 'resolve_delivery_transport'),
    'resolve_shutdown_watchdog_delay': ('gateway.shutdown_watchdog', 'resolve_shutdown_watchdog_delay'),
    'start_loop_liveness_watchdog': ('gateway.shutdown_watchdog', 'start_loop_liveness_watchdog'),
    't': ('agent.i18n', 't'),
    'utf16_len': ('gateway.platforms.base', 'utf16_len'),
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
