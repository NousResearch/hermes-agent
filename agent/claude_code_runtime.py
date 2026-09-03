"""Claude Code runtime — hand a turn to a ``claude -p`` subprocess (#25267).

Mirror of :mod:`agent.codex_runtime.run_codex_app_server_turn` for the
``claude_code`` api_mode: the official Claude CLI owns the model loop and its
native tools, Hermes' tools reach it over stdio MCP, and the CLI's stream-json
events are projected back into Hermes' ``messages`` list so memory review,
session persistence, and every gateway keep working unchanged.

Sites that string-compare ``"codex_app_server"`` and what this runtime does
about each (grep ``codex_app_server`` to audit):

  MIRRORED
    agent/conversation_loop.py   dispatch           -> ``elif api_mode == "claude_code"``
    run_agent.py                 _run_*_turn         -> ``_run_claude_code_turn`` forwarder
    run_agent.py                 interrupt()         -> forwards to ``_claude_code_session.request_interrupt``
    run_agent.py                 close()             -> closes ``_claude_code_session``
    agent/agent_init.py          valid api_mode set  -> ``"claude_code"`` added
    agent/agent_init.py          checkpoint refusal  -> Claude Code compacts its own context, same as codex
    agent/agent_init.py          client construction -> no OpenAI client; ``agent.client = None``
    agent/turn_context.py        preflight compress  -> skipped (native compaction; == ``codex_app_server_auto: native``)
    agent/turn_context.py        api-bytes stamp     -> skipped (the CLI never sees ``api_messages``)
    agent/conversation_compression.py compress_context -> no-op boundary (native compaction only)
    agent/background_review.py   _run_review_in_thread -> refused on claude_code for EVERY caller
                                                        (automatic, /refine, CLI command): the fork
                                                        would start a second `claude` that cannot
                                                        reach the `memory` / `skill_manage` tools
                                                        (they are _AGENT_LOOP_TOOLS, not MCP-exposable)
    tui_gateway/server.py        image_routing       -> forces text mode for codex_app_server; claude
                                                        images are not forwarded either (see
                                                        _coerce_input_text) so the same downgrade applies
    hermes_cli/runtime_provider.py _VALID_API_MODES  -> ``"claude_code"`` added; ``claude-code`` provider resolves
                                                        with no base_url / api_key

  NOT MIRRORED (deliberately)
    run_agent.py                 steer()             -> codex has ``turn/steer``; stream-json has no
                                                        equivalent. Falls through to the default
                                                        next-turn queue.
    agent/conversation_compression.py _compress_context_via_codex_app_server
                                                     -> ``compression.codex_app_server_auto: hermes``
                                                        has no analogue; Claude Code exposes no compact RPC.
    agent/native_compaction.py                       -> codex-only accounting for the above.
    hermes_cli/codex_runtime_switch.py, commands.py, banner.py, cli.py /codex-runtime,
    gateway/slash_commands.py, codex_runtime_plugin_migration.py
                                                     -> codex-specific opt-in UX (``model.openai_runtime``).
                                                        claude_code is selected by provider, not a switch.
    hermes_cli/config_defaults.py, gateway/run.py    -> ``compression.codex_app_server_auto`` config key.
                                                        claude_code behaves as the ``native`` value of
                                                        that knob, permanently: the CLI auto-compacts
                                                        its own context, Hermes never summarizes the
                                                        local mirror, and there is no ``hermes``/``off``
                                                        analogue because the CLI exposes no compact RPC.
    agent/transports/hermes_tools_mcp_server.py      -> docstring mentions only; shared as-is.

Known gap shared with the codex runtime: Hermes tools that need the live
agent loop (``memory``, ``delegate_task``, ``session_search``, ``todo``) are
not exposed over MCP.

TODO: background memory/skill review is skipped on this runtime (the guard
lives in ``agent/background_review._run_review_in_thread`` so every caller —
automatic, ``/refine``, CLI command — converges on it). The follow-up that
would restore it is a stateless ``memory`` MCP tool backed by a fresh
``MemoryStore`` loaded from disk per call, with the parent reloading its
store after each turn.

Isolation contract (HIGH-2 in review): the child is a Hermes-owned Claude
Code — ``CLAUDE_CONFIG_DIR=$HERMES_HOME/claude-code``, cwd
``$HERMES_HOME/claude-code/workspace`` (or ``claude_code.cwd``),
``--setting-sources ""`` + ``--settings $HERMES_HOME/claude-code/settings.json``
(deny list), ``--strict-mcp-config``, auto-memory off, and its own
setup-token credential in ``$CLAUDE_CODE_OAUTH_TOKEN``. The user's
``~/.claude`` (hooks, plugins, memory, transcripts, interactive login) is never
read or written.
"""

from __future__ import annotations

import atexit
import glob
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def combined_system_prompt(agent) -> str:
    """The system prompt a chat_completions provider would receive.

    ``conversation_loop`` sends ``_cached_system_prompt`` followed by
    ``ephemeral_system_prompt`` (the gateway's per-request ``system`` message —
    for the api_server path that is the caller's whole prompt) joined by a
    blank line. The child must see exactly that, so the same combination is
    baked into ``--append-system-prompt-file``.
    """
    base = getattr(agent, "_cached_system_prompt", None) or ""
    ephemeral = getattr(agent, "ephemeral_system_prompt", None) or ""
    if ephemeral:
        return (base + "\n\n" + ephemeral).strip()
    return base.strip()


# ---------------------------------------------------------------------------
# Warm-process registry
#
# api_server builds a fresh AIAgent per request (stateless by design; history
# reloads from state.db). Caching the session on the agent instance therefore
# spawned a new `claude` + warm-up per request and lost the prompt-cache
# warmth the runtime exists for. Sessions are keyed by the Hermes session id
# (the same key the CLI session id is derived from) and shared across agent
# instances. Turns are serialised per session; idle processes are evicted.
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_IDLE_SECONDS = 600.0
DEFAULT_MAX_SESSIONS = 8
_REGISTRY_SWEEP_INTERVAL = 30.0
#: Respawn rate guard: more than this many prompt-driven respawns inside the
#: window means the caller's system prompt changes on (nearly) every request,
#: which defeats the warm process and the prompt cache.
_RESPAWN_WARN_COUNT = 3
_RESPAWN_WARN_WINDOW = 60.0
#: Temp files older than this in the config dir are leftovers from a crashed
#: or pre-registry process and are removed on startup.
_STALE_TEMP_AGE_SECONDS = 24 * 3600
#: How long a second request for the same session waits for the in-flight
#: turn before giving up with a clear error.
_TURN_LOCK_WAIT_SECONDS = 15.0


@dataclass
class _RegistryEntry:
    session: Any = None
    turn_lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)
    # Construction happens OUTSIDE the registry lock (config + session-map
    # reads); ``ready`` gates readers until ``session`` is populated.
    ready: threading.Event = field(default_factory=threading.Event)
    respawn_times: list = field(default_factory=list)
    respawn_warned: bool = False


_REGISTRY: Dict[str, _RegistryEntry] = {}
_REGISTRY_LOCK = threading.Lock()
_SWEEPER_STARTED = False


def _registry_key(agent) -> Optional[str]:
    sid = str(getattr(agent, "session_id", "") or "").strip()
    return sid or None


def _start_sweeper_locked() -> None:
    global _SWEEPER_STARTED
    if _SWEEPER_STARTED:
        return
    _SWEEPER_STARTED = True

    def _sweep_forever() -> None:
        while True:
            time.sleep(_REGISTRY_SWEEP_INTERVAL)
            try:
                sweep_idle_sessions(_idle_timeout_from_config())
            except Exception:
                logger.debug("claude-code registry sweep failed", exc_info=True)

    threading.Thread(target=_sweep_forever, daemon=True, name="claude-code-registry").start()


def _max_sessions_from_config() -> int:
    try:
        value = int(_claude_code_config().get("max_sessions") or DEFAULT_MAX_SESSIONS)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_SESSIONS
    return max(1, value)


def _idle_timeout_from_config() -> float:
    try:
        return float(_claude_code_config().get("idle_timeout") or DEFAULT_REGISTRY_IDLE_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_REGISTRY_IDLE_SECONDS


def sweep_idle_sessions(idle_seconds: float, *, now: Optional[float] = None) -> int:
    """Close and drop every registered session idle for ``idle_seconds``.
    Sessions with a turn in flight are never evicted. Returns the count."""
    now = time.monotonic() if now is None else now
    victims: list[tuple[str, _RegistryEntry]] = []
    with _REGISTRY_LOCK:
        for key, entry in list(_REGISTRY.items()):
            if now - entry.last_used < idle_seconds:
                continue
            if not entry.turn_lock.acquire(blocking=False):
                continue  # a turn is running
            try:
                victims.append((key, _REGISTRY.pop(key)))
            finally:
                entry.turn_lock.release()
    for key, entry in victims:
        logger.info("claude-code: evicting idle session %s (pid %s)", key[:12], entry.session.pid)
        try:
            entry.session.close()
        except Exception:
            pass
    return len(victims)


def evict_session(key: Optional[str]) -> None:
    """Drop and close the registered session for ``key`` (retire / close)."""
    if not key:
        return
    with _REGISTRY_LOCK:
        entry = _REGISTRY.pop(key, None)
    if entry is not None and entry.session is not None:
        try:
            entry.session.close()
        except Exception:
            pass


def _shutdown_registry() -> None:
    """atexit: close every warm child and unlink its temp prompt/MCP files."""
    with _REGISTRY_LOCK:
        keys = list(_REGISTRY)
    for key in keys:
        evict_session(key)


atexit.register(_shutdown_registry)

_STALE_TEMP_PRUNED = False


def prune_stale_temp_files(config_dir: str, *, max_age: float = _STALE_TEMP_AGE_SECONDS) -> int:
    """Remove ``system-prompt-*.md`` / ``hermes-claude-mcp-*.json`` leftovers
    older than ``max_age`` seconds (crashed processes, pre-registry code).
    Files younger than that may belong to a live child and are kept."""
    removed = 0
    cutoff = time.time() - max_age
    for pattern in ("system-prompt-*.md", "hermes-claude-mcp-*.json"):
        for path in glob.glob(os.path.join(config_dir, pattern)):
            try:
                if os.path.getmtime(path) < cutoff:
                    os.unlink(path)
                    removed += 1
            except OSError:
                pass
    if removed:
        logger.info("claude-code: removed %d stale temp file(s) from %s", removed, config_dir)
    return removed


def _prune_stale_temp_files_once() -> None:
    global _STALE_TEMP_PRUNED
    if _STALE_TEMP_PRUNED:
        return
    _STALE_TEMP_PRUNED = True
    try:
        from agent.transports.claude_code_session import claude_code_home

        prune_stale_temp_files(claude_code_home())
    except Exception:
        logger.debug("claude-code: stale temp prune failed", exc_info=True)


def registered_session_count() -> int:
    with _REGISTRY_LOCK:
        return len(_REGISTRY)


class TurnInFlightError(RuntimeError):
    """A second request for a session arrived while a turn was running."""


def _evict_lru_locked(max_sessions: int) -> list[tuple[str, _RegistryEntry]]:
    """Under ``_REGISTRY_LOCK``: pop the least-recently-used NOT-in-flight
    entries until there is room for one more. Returns them for closing."""
    victims: list[tuple[str, _RegistryEntry]] = []
    while len(_REGISTRY) >= max_sessions:
        candidates = sorted(_REGISTRY.items(), key=lambda kv: kv[1].last_used)
        for key, entry in candidates:
            if not entry.ready.is_set():
                continue
            if entry.turn_lock.acquire(blocking=False):
                entry.turn_lock.release()
                victims.append((key, _REGISTRY.pop(key)))
                break
        else:
            break  # every entry is mid-turn; allow temporary overflow
    return victims


def _acquire_entry(agent) -> tuple[_RegistryEntry, bool]:
    """Return ``(entry, created)`` for the agent's session, holding its turn
    lock. Rebuilds a closed/dead session in place; agents without a session
    id get a private, unregistered entry. Session construction runs outside
    the registry lock (a placeholder entry + ``ready`` event serialises only
    the readers of THAT key)."""
    key = _registry_key(agent)
    if key is None:
        entry = _RegistryEntry(session=_build_session(agent))
        entry.ready.set()
        entry.turn_lock.acquire()
        return entry, True
    created = False
    victims: list[tuple[str, _RegistryEntry]] = []
    with _REGISTRY_LOCK:
        _start_sweeper_locked()
        entry = _REGISTRY.get(key)
        if (
            entry is not None
            and entry.ready.is_set()
            and (entry.session is None or getattr(entry.session, "_closed", False))
        ):
            _REGISTRY.pop(key, None)
            entry = None
        if entry is None:
            victims = _evict_lru_locked(_max_sessions_from_config())
            entry = _RegistryEntry()
            _REGISTRY[key] = entry
            created = True
    for victim_key, victim in victims:
        logger.info(
            "claude-code: registry full (max_sessions=%d); evicting LRU session %s (pid %s)",
            _max_sessions_from_config(), victim_key[:12], getattr(victim.session, "pid", None),
        )
        try:
            victim.session.close()
        except Exception:
            pass
    if created:
        try:
            entry.session = _build_session(agent)
        except BaseException:
            with _REGISTRY_LOCK:
                if _REGISTRY.get(key) is entry:
                    _REGISTRY.pop(key, None)
            entry.ready.set()
            raise
        entry.ready.set()
    elif not entry.ready.wait(timeout=_TURN_LOCK_WAIT_SECONDS) or entry.session is None:
        raise TurnInFlightError(
            f"session {key[:12]} is still being set up by another request; try again"
        )
    if not entry.turn_lock.acquire(timeout=_TURN_LOCK_WAIT_SECONDS):
        raise TurnInFlightError(
            f"another turn is still running for session {key[:12]}; try again"
        )
    if not created:
        # The turn we waited on may have retired this entry (evicted + closed
        # under its lock). Never hand out a dead session: drop our reference
        # and acquire again, which rebuilds it.
        with _REGISTRY_LOCK:
            stale = _REGISTRY.get(key) is not entry
        if stale or entry.session is None or getattr(entry.session, "_closed", False):
            entry.turn_lock.release()
            return _acquire_entry(agent)
    entry.last_used = time.monotonic()
    return entry, created


def _note_respawn(entry: _RegistryEntry, key: Optional[str]) -> None:
    """Rate guard for prompt-driven respawns (see _RESPAWN_WARN_COUNT)."""
    now = time.monotonic()
    entry.respawn_times = [t for t in entry.respawn_times if now - t < _RESPAWN_WARN_WINDOW] + [now]
    if len(entry.respawn_times) > _RESPAWN_WARN_COUNT and not entry.respawn_warned:
        entry.respawn_warned = True
        logger.warning(
            "claude-code: session %s respawned %d times in %.0fs because its system "
            "prompt changes every request; make the prompt stable (no timestamps / "
            "per-request content) or the warm process and prompt cache are lost",
            (key or "-")[:12], len(entry.respawn_times), _RESPAWN_WARN_WINDOW,
        )


def _claude_code_config() -> Dict[str, Any]:
    """The optional ``claude_code:`` block from config.yaml.

    Keys (all optional)::

        claude_code:
          binary: claude            # path to the CLI
          oauth_token_env: CLAUDE_CODE_OAUTH_TOKEN  # env var holding `claude setup-token`
          cwd: ""                   # child working dir (default $HERMES_HOME/claude-code/workspace)
          deny: []                  # permissions.deny rules for the first-written settings.json
          permission_mode: ""       # override: default|acceptEdits|plan|dontAsk|bypassPermissions
          allowed_tools: []         # override the native-tool allowlist
          expose_hermes_tools: true # register hermes-tools MCP server
          extra_args: []            # appended verbatim to the claude command line
          resume: true              # --resume the CLI transcript of a resumed Hermes session
          turn_timeout: 600         # whole-turn bound (seconds)
          silence_timeout: 300      # max silence between two CLI events inside a turn
          idle_timeout: 600         # evict a warm `claude` process idle this long (registry)
          max_sessions: 8           # warm processes kept at once (LRU eviction beyond this)
    """
    try:
        from hermes_cli.config import load_config

        block = load_config().get("claude_code")
    except Exception:
        block = None
    return dict(block) if isinstance(block, dict) else {}


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _record_claude_code_usage(agent, turn) -> Dict[str, Any]:
    """Translate the CLI ``result.usage`` block into Hermes accounting.

    Claude Code reports Anthropic-shaped usage: ``input_tokens``,
    ``output_tokens``, ``cache_read_input_tokens``,
    ``cache_creation_input_tokens``. Hermes' canonical prompt bucket is
    uncached input + cache read + cache write. Subscription usage has no
    per-token price, so cost is recorded as included.
    """
    agent.session_api_calls += 1
    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if getattr(agent, "_session_db", None) and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.queue_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug("claude-code api-call persistence failed: %s", exc)
        return {}

    from agent.usage_pricing import CanonicalUsage

    canonical = CanonicalUsage(
        input_tokens=_coerce_int(usage.get("input_tokens")),
        output_tokens=_coerce_int(usage.get("output_tokens")),
        cache_read_tokens=_coerce_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_coerce_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical.prompt_tokens
    completion_tokens = canonical.output_tokens
    total_tokens = canonical.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical.input_tokens,
        "output_tokens": canonical.output_tokens,
        "cache_read_tokens": canonical.cache_read_tokens,
        "cache_write_tokens": canonical.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
            window = getattr(turn, "model_context_window", None)
            if isinstance(window, int) and window > 0:
                compressor.context_length = window
        except Exception:
            logger.debug("claude-code usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical.input_tokens
    agent.session_output_tokens += canonical.output_tokens
    agent.session_cache_read_tokens += canonical.cache_read_tokens
    agent.session_cache_write_tokens += canonical.cache_write_tokens
    agent.session_cost_status = "included"
    agent.session_cost_source = "claude-code-subscription"

    if getattr(agent, "_session_db", None) and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id,
                input_tokens=canonical.input_tokens,
                output_tokens=canonical.output_tokens,
                cache_read_tokens=canonical.cache_read_tokens,
                cache_write_tokens=canonical.cache_write_tokens,
                reasoning_tokens=0,
                estimated_cost_usd=None,
                cost_status="included",
                cost_source="claude-code-subscription",
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included",
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug("claude-code token persistence failed: %s", exc)

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": None,
        "cost_status": "included",
        "cost_source": "claude-code-subscription",
    }


def make_claude_code_event_bridge(agent) -> Callable[[dict], None]:
    """Wire session events into Hermes' gateway UI callbacks.

    Same contract as ``make_codex_app_server_event_bridge`` (#33200): text
    deltas -> ``_fire_stream_delta``, thinking -> ``_fire_reasoning_delta``,
    tool lifecycle -> ``tool_progress_callback`` / ``tool_start_callback`` /
    ``tool_complete_callback``, commentary before a tool call ->
    ``_emit_interim_assistant_message``. Every callback is guarded so a buggy
    display hook can never tear down the turn.
    """

    def _call(name: str, *args: Any, **kwargs: Any) -> None:
        fn = getattr(agent, name, None)
        if fn is None:
            return
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.debug("%s raised", name, exc_info=True)

    def on_event(event: dict) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("kind")
        if kind == "text_delta":
            _call("_fire_stream_delta", event.get("text") or "")
        elif kind == "reasoning_delta":
            _call("_fire_reasoning_delta", event.get("text") or "")
        elif kind == "tool_started":
            name = event.get("name") or "tool"
            args = event.get("args") or {}
            _call("tool_progress_callback", "tool.started", name, event.get("preview"), args)
            _call("tool_start_callback", event.get("call_id"), name, args)
        elif kind == "tool_completed":
            name = event.get("name") or "tool"
            _call(
                "tool_progress_callback", "tool.completed", name, None, None,
                duration=event.get("duration"), is_error=bool(event.get("is_error")),
                result=event.get("result"),
            )
            _call(
                "tool_complete_callback", event.get("call_id"), name,
                event.get("args") or {}, event.get("result"),
            )
        elif kind == "assistant_message":
            if not getattr(agent, "show_commentary", True):
                return
            text = event.get("text") or ""
            if text.strip():
                _call("_emit_interim_assistant_message", {"role": "assistant", "content": text})
        elif kind == "status":
            _call("_emit_status", event.get("text") or "")

    return on_event


# Namespace for deriving the CLI session id from the Hermes session id. A
# deterministic id means a resumed Hermes session can ``--resume`` its CLI
# transcript without persisting anything extra (mirrors how the codex runtime
# only *returns* ``codex_thread_id`` — nothing downstream stores it).
_CLAUDE_SESSION_NAMESPACE = uuid.UUID("6f1c0d2e-7b3a-4c8e-9a5d-2f4b8c1e7d90")


def _claude_session_id_for(agent) -> str:
    hermes_sid = str(getattr(agent, "session_id", "") or "").strip()
    if not hermes_sid:
        return str(uuid.uuid4())
    return str(uuid.uuid5(_CLAUDE_SESSION_NAMESPACE, f"hermes:{hermes_sid}"))


def _build_session(agent):
    from agent.transports.claude_code_session import ClaudeCodeSession

    _prune_stale_temp_files_once()

    from agent.transports.claude_code_session import DEFAULT_OAUTH_TOKEN_ENV

    cfg = _claude_code_config()
    # Dedicated workspace by default — NOT the Hermes process cwd ($HOME for
    # the gateway), so CLAUDE.md discovery and transcript slugs never key off
    # wherever Hermes happened to be launched.
    cwd = str(cfg.get("cwd") or "").strip() or None

    # ``--yolo`` / ``approvals.mode: off`` / the gateway /yolo toggle are the
    # user's explicit "don't gate me" — the only route to bypassPermissions
    # besides tools.terminal.security_mode=unrestricted.
    security_mode = os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto")
    try:
        from tools.approval import is_approval_bypass_active

        if is_approval_bypass_active():
            security_mode = "unrestricted"
    except Exception:
        logger.debug("claude-code: approval-bypass lookup failed", exc_info=True)

    allowed = cfg.get("allowed_tools")
    extra_args = cfg.get("extra_args")
    deny = cfg.get("deny")
    # Approval prompt: defer to Hermes' standard flow when the CLI thread
    # installed one (same source the codex runtime uses). Gateway / cron
    # contexts have none -> gated tools are denied with a message.
    approval_callback = _approval_callback()
    hermes_sid = str(getattr(agent, "session_id", "") or "").strip() or None
    return ClaudeCodeSession(
        cwd=cwd,
        oauth_token_env=str(cfg.get("oauth_token_env") or DEFAULT_OAUTH_TOKEN_ENV),
        deny_rules=[str(r) for r in deny] if isinstance(deny, list) and deny else None,
        session_id=_claude_session_id_for(agent),
        session_key=hermes_sid,
        resume=bool(cfg.get("resume", True)),
        approval_callback=approval_callback,
        claude_bin=str(cfg.get("binary") or "claude"),
        model=getattr(agent, "model", None),
        security_mode=security_mode,
        permission_mode=str(cfg.get("permission_mode") or "") or None,
        allowed_tools=[str(t) for t in allowed] if isinstance(allowed, list) else None,
        system_prompt=combined_system_prompt(agent),
        expose_hermes_tools=bool(cfg.get("expose_hermes_tools", True)),
        extra_args=[str(a) for a in extra_args] if isinstance(extra_args, list) else None,
        on_event=make_claude_code_event_bridge(agent),
    )


def run_claude_code_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Run one turn through the Claude Code subprocess.

    Called from run_conversation() when ``agent.api_mode == "claude_code"``.
    Returns the same dict shape as the chat_completions path.
    """
    if getattr(agent, "compression_checkpoint_required", False) is True:
        from agent.conversation_compression import _checkpoint_blocked

        raise _checkpoint_blocked(
            "claude_code owns the authoritative context and compacts it "
            "without a truthful pre-compaction transcript boundary"
        )

    cfg = _claude_code_config()
    registry_key = _registry_key(agent)
    try:
        entry, created = _acquire_entry(agent)
    except TurnInFlightError as exc:
        return {
            "final_response": str(exc),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": False,
            "error": str(exc),
        }
    session = entry.session
    # The registry hands the same process to every AIAgent instance of a
    # session; point its UI/approval hooks at THIS instance for the turn and
    # expose it for interrupt()/close() forwarding.
    session.rebind(
        on_event=make_claude_code_event_bridge(agent),
        approval_callback=_approval_callback(),
    )
    agent._claude_code_session = session
    try:
        # The system prompt (Hermes' cached prompt + the gateway's ephemeral
        # prompt) is baked into the process at spawn. If either changed —
        # memory rebuilt, /model, a new per-request system message —
        # respawn so the CLI sees the new one (same CLI session, resumed).
        wanted_prompt = combined_system_prompt(agent)
        if not created and session.needs_respawn(wanted_prompt):
            logger.info("claude-code: system prompt changed; respawning session")
            _note_respawn(entry, registry_key)
            try:
                session.restart(system_prompt=wanted_prompt)
            except Exception:
                logger.warning("claude-code respawn failed; rebuilding session", exc_info=True)
                try:
                    session.close()
                except Exception:
                    pass
                session = _build_session(agent)
                entry.session = session
                agent._claude_code_session = session

        # NOTE: the user message is ALREADY in ``messages`` (appended by
        # run_conversation before dispatch). Do not append it again.
        try:
            session.ensure_started()
            turn = session.run_turn(
                user_input=user_message,
                turn_timeout=float(cfg.get("turn_timeout") or 600.0),
                idle_timeout=float(cfg.get("silence_timeout") or 300.0),
            )
        except Exception as exc:
            logger.exception("claude-code turn failed")
            evict_session(registry_key)
            try:
                session.close()
            except Exception:
                pass
            agent._claude_code_session = None
            _user_interrupted = bool(getattr(agent, "_interrupt_requested", False))
            _interrupt_message = (
                getattr(agent, "_interrupt_message", None) if _user_interrupted else None
            )
            if _user_interrupted:
                agent.clear_interrupt()
            return {
                "final_response": f"Claude Code turn failed: {exc}",
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "interrupted": _user_interrupted,
                **({"interrupt_message": _interrupt_message} if _interrupt_message else {}),
                "error": str(exc),
            }
        entry.last_used = time.monotonic()
        if getattr(turn, "should_retire", False):
            # Retire while still holding the turn lock: a request waiting on
            # this session must wake to a fresh process, not to one we are
            # about to close underneath it.
            logger.warning("claude-code session retired (turn error: %s)", turn.error)
            evict_session(registry_key)
            try:
                session.close()
            except Exception:
                pass
            agent._claude_code_session = None
    finally:
        entry.turn_lock.release()
    return _finish_turn(
        agent, turn, cfg,
        registry_key=registry_key, session=session,
        user_message=user_message, original_user_message=original_user_message,
        messages=messages, should_review_memory=should_review_memory,
    )


def _approval_callback():
    try:
        from tools.terminal_tool import _get_approval_callback

        return _get_approval_callback()
    except Exception:
        return None


def _finish_turn(
    agent, turn, cfg, *, registry_key, session, user_message, original_user_message,
    messages, should_review_memory,
) -> Dict[str, Any]:
    """Post-turn bookkeeping shared with the codex runtime shape."""

    _user_interrupted = bool(
        turn.interrupted and getattr(agent, "_interrupt_requested", False)
    )
    _interrupt_message = (
        getattr(agent, "_interrupt_message", None) if _user_interrupted else None
    )
    if _user_interrupted:
        agent.clear_interrupt()

    if turn.projected_messages:
        from agent.message_metadata import append_message

        for projected in turn.projected_messages:
            append_message(messages, projected)
        # Early-return path: flush the projected rows ourselves (idempotent
        # via _DB_PERSISTED_MARKER; the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                flush_ok = agent._flush_messages_to_session_db(messages)
            except Exception:
                flush_ok = False
                logger.warning("claude-code projected-message flush failed", exc_info=True)
            if flush_ok is False:
                logger.warning(
                    "claude-code turn was delivered but could NOT be persisted "
                    "to the session DB (session=%s)",
                    getattr(agent, "session_id", None),
                )

    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_code_usage(agent, turn)

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review is refused inside background_review._run_review_in_thread
    # for this api_mode (every caller converges there); nothing to gate here.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    final_text = turn.final_text
    if turn.error and not final_text:
        final_text = turn.error
    return {
        "final_response": final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "interrupted": _user_interrupted,
        **({"interrupt_message": _interrupt_message} if _interrupt_message else {}),
        "error": turn.error,
        "agent_persisted": True,
        "claude_session_id": turn.session_id,
        **usage_result,
    }
