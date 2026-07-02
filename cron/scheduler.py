"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import atexit
import concurrent.futures
import contextvars
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports BEFORE repo-level imports.
# Without this, standalone invocations (e.g. after `hermes update` reloads
# the module) fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import load_config, _expand_env_vars
from hermes_cli.fallback_config import get_fallback_chain
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)


def _summarize_cron_failure_for_delivery(job: dict, error: str | None) -> str:
    """Return a compact one-line failure message for chat delivery.

    Full details stay in the cron output directory and the logs. Chat should
    show the operator what broke without dumping provider JSON, retry noise, or
    stack traces into the delivery channel.
    """
    job_name = job.get("name") or job.get("id") or "cron job"
    text = (error or "unknown error").strip()
    lower = text.lower()

    # Provider/API failures are the common noisy path. Keep these short.
    if "429" in text or "rate limit" in lower or "usage limit" in lower:
        reason = "rate limit"
        if "weekly usage limit" in lower:
            reason = "weekly usage limit"
        elif "quota" in lower:
            reason = "quota limit"
        return (
            f"⚠️ Cron '{job_name}' failed: provider {reason}. "
            "Fallback chain was exhausted or unavailable. "
            "Full details saved in cron output."
        )

    if "readtimeout" in lower or "timed out" in lower or "timeout" in lower:
        return (
            f"⚠️ Cron '{job_name}' failed: provider timeout. "
            "Fallback chain was exhausted or unavailable. "
            "Full details saved in cron output."
        )

    # Match authentication/authorization wording at a word boundary and the
    # 401/403 status codes as whole tokens, so "oauth", "4015" and similar do
    # not trip a misleading auth message.
    if re.search(r"authenticat|authoriz", lower) or re.search(r"\b(401|403)\b", text):
        return (
            f"⚠️ Cron '{job_name}' failed: provider authentication error. "
            "Full details saved in cron output."
        )

    # Strip common exception wrappers and collapse provider payloads. Bound
    # the input first so a multi-KB provider blob cannot slow the
    # substitutions.
    cleaned = re.sub(
        r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*",
        "", text[:2000],
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 180:
        cleaned = cleaned[:177].rstrip() + "..."
    return f"⚠️ Cron '{job_name}' failed: {cleaned}"


def _format_model_provider_line(model: Optional[str], provider: Optional[str]) -> str:
    """Render the ``**Model:** {model} ({provider})`` header line (#56791).

    Empty/missing inputs produce empty strings; callers gate the output on
    truthiness so headers stay backwards-compatible for jobs with no resolved
    model/provider yet (e.g. font-end-of-build races between provider
    selection and the finally-block that emits the markdown).
    """
    m = (model or "").strip()
    p = (provider or "").strip()
    if not m and not p:
        return ""
    if m and p:
        return f"**Model:** {m} ({p})\n"
    if m:
        return f"**Model:** {m}\n"
    return f"**Provider:** {p}\n"


class CronPromptInjectionBlocked(Exception):
    """Raised by _build_job_prompt when the fully-assembled prompt trips the
    injection scanner. Caught in run_job so the operator sees a clean
    "job blocked" delivery instead of the scheduler crashing.

    Assembled-prompt scanning (including loaded skill content) plugs the
    gap from #3968: create-time scanning only covers the user-supplied
    prompt field; skill content loaded at runtime was never scanned, so a
    malicious skill could carry an injection payload that reached the
    non-interactive (auto-approve) cron agent.
    """


def _resolve_cron_disabled_toolsets(cfg: dict) -> list[str]:
    """Toolsets a cron-spawned agent must never receive.

    Three protected toolsets are always disabled in cron context:
      - ``cronjob`` — would let a cron-spawned agent schedule more cron jobs
      - ``messaging`` — interactive, needs a live gateway session
      - ``clarify`` — interactive, blocks waiting for user input

    User-level ``agent.disabled_toolsets`` from config.yaml is layered on top
    so per-job ``enabled_toolsets`` cannot bypass policy that applies to
    ordinary agent runs (#25752 — LLM-supplied enabled_toolsets was widening
    past config.yaml's denylist).
    """
    disabled = ["cronjob", "messaging", "clarify"]
    agent_cfg = (cfg or {}).get("agent") or {}
    user_disabled = agent_cfg.get("disabled_toolsets") or []
    for name in user_disabled:
        name = str(name).strip()
        if name and name not in disabled:
            disabled.append(name)
    return disabled


def _merge_mcp_into_per_job_toolsets(per_job: list[str], cfg: dict) -> list[str]:
    """Layer enabled MCP servers onto a per-job ``enabled_toolsets`` allowlist.

    A per-job list scopes the *native* toolsets, but on its own it silently
    drops every MCP server: ``discover_mcp_tools()`` registers the tools into
    the global registry, yet ``get_tool_definitions(enabled_toolsets=...)``
    only keeps toolsets named in the list. The agent then rejects every
    ``mcp_*`` call with "Unknown tool". This restores parity with
    ``_get_platform_tools`` MCP semantics:

      * ``no_mcp`` sentinel present  -> no MCP servers (sentinel stripped)
      * one or more MCP server names already listed -> treat as an allowlist,
        add nothing further (the user named exactly the servers they want)
      * otherwise -> union in every globally-enabled MCP server
    """
    result = [t for t in per_job if t != "no_mcp"]
    if "no_mcp" in per_job:
        return result
    # lazy import: avoid heavy hermes_cli import at cron module load (matches
    # _resolve_cron_enabled_toolsets' fallback) and share one MCP-membership
    # computation with the gateway/CLI platform resolver.
    from hermes_cli.tools_config import enabled_mcp_server_names
    enabled_mcp = enabled_mcp_server_names(cfg)
    if set(result) & enabled_mcp:
        return result
    for name in sorted(enabled_mcp):
        if name not in result:
            result.append(name)
    return result


def _resolve_cron_enabled_toolsets(job: dict, cfg: dict) -> list[str] | None:
    """Resolve the toolset list for a cron job.

    Precedence:
    1. Per-job ``enabled_toolsets`` (set via ``cronjob`` tool on create/update).
       Keeps the agent's job-scoped toolset override intact — #6130. Enabled
       MCP servers are layered on per ``_merge_mcp_into_per_job_toolsets`` so a
       native-toolset allowlist does not silently strip MCP tools.
    2. Per-platform ``hermes tools`` config for the ``cron`` platform.
       Mirrors gateway behavior (``_get_platform_tools(cfg, platform_key)``)
       so users can gate cron toolsets globally without recreating every job.
    3. ``None`` on any lookup failure — AIAgent loads the full default set
       (legacy behavior before this change, preserved as the safety net).

    _DEFAULT_OFF_TOOLSETS ({moa, homeassistant, rl}) are removed by
    ``_get_platform_tools`` for unconfigured platforms, so fresh installs
    get cron WITHOUT ``moa`` by default (issue reported by Norbert —
    surprise $4.63 run).
    """
    per_job = job.get("enabled_toolsets")
    if per_job:
        return _merge_mcp_into_per_job_toolsets(list(per_job), cfg or {})
    try:
        from hermes_cli.tools_config import _get_platform_tools  # lazy: avoid heavy import at cron module load
        return sorted(_get_platform_tools(cfg or {}, "cron"))
    except Exception as exc:
        logger.warning(
            "Cron toolset resolution failed, falling back to full default toolset: %s",
            exc,
        )
        return None

# Valid delivery platforms — used to validate user-supplied platform names
# in cron delivery targets, preventing env var enumeration via crafted names.
_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "mattermost", "homeassistant", "dingtalk", "feishu",
    "wecom", "wecom_callback", "weixin", "sms", "email", "webhook", "bluebubbles",
    "qqbot", "yuanbao",
})

# Platforms that support a configured cron/notification home target, mapped to
# the environment variable used by gateway setup/runtime config.
_HOME_TARGET_ENV_VARS = {
    "matrix": "MATRIX_HOME_ROOM",
    "telegram": "TELEGRAM_HOME_CHANNEL",
    "discord": "DISCORD_HOME_CHANNEL",
    "slack": "SLACK_HOME_CHANNEL",
    "signal": "SIGNAL_HOME_CHANNEL",
    "mattermost": "MATTERMOST_HOME_CHANNEL",
    "sms": "SMS_HOME_CHANNEL",
    "email": "EMAIL_HOME_ADDRESS",
    "dingtalk": "DINGTALK_HOME_CHANNEL",
    "feishu": "FEISHU_HOME_CHANNEL",
    "wecom": "WECOM_HOME_CHANNEL",
    "weixin": "WEIXIN_HOME_CHANNEL",
    "bluebubbles": "BLUEBUBBLES_HOME_CHANNEL",
    "qqbot": "QQBOT_HOME_CHANNEL",
    "whatsapp": "WHATSAPP_HOME_CHANNEL",
    "whatsapp_cloud": "WHATSAPP_CLOUD_HOME_CHANNEL",
}

# Legacy env var names kept for back-compat.  Each entry is the current
# primary env var → the previous name.  _get_home_target_chat_id falls
# back to the legacy name if the primary is unset, so users who set the
# old name before the rename keep working until they migrate.
_LEGACY_HOME_TARGET_ENV_VARS = {
    "QQBOT_HOME_CHANNEL": "QQ_HOME_CHANNEL",
}

from cron.jobs import get_due_jobs, mark_job_run, save_job_output, advance_next_run, claim_dispatch

# Sentinel: when a cron agent has nothing new to report, it can start its
# response with this marker to suppress delivery.  Output is still saved
# locally for audit.
SILENT_MARKER = "[SILENT]"

# Canonical silence tokens recognized in cron output.  Cron's contract is
# intentionally looser than the gateway's exact-whole-response rule: the cron
# system prompt *instructs* the agent to emit "[SILENT]", and real agents often
# bracket it with a short note or trailing newline.  We therefore suppress when
# a marker is the entire response OR appears as its own first/last line — but
# NOT when a token merely appears mid-sentence in a genuine report (e.g.
# "I considered staying [SILENT] but here is the summary…" must deliver).
_CRON_SILENCE_TOKENS = frozenset({"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"})


def _is_cron_silence_response(text: str) -> bool:
    """Return True when a cron final response should suppress delivery.

    Recognizes the bracketed ``[SILENT]`` sentinel (whole-response, first line,
    or last line) plus the bracketless ``SILENT`` / ``NO_REPLY`` / ``NO REPLY``
    variants the model emits when it drops the brackets (#51438, #46917).
    Whitespace-trimmed and case-insensitive.  A token buried mid-sentence is
    treated as real content and delivered.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False

    def _is_token(line: str) -> bool:
        return " ".join(line.strip().upper().split()) in _CRON_SILENCE_TOKENS

    # Whole response is exactly a token.
    if _is_token(stripped):
        return True
    # Marker on its own first or last line (trailing/leading note on a
    # separate line — e.g. "2 deals filtered\n\n[SILENT]").
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if lines and (_is_token(lines[0]) or _is_token(lines[-1])):
        return True
    # Bracketed sentinel used as a same-line prefix — the documented cron
    # pattern "[SILENT] No changes detected".  Restricted to the bracketed
    # form so a bare word like "Silent retry succeeded" is NOT swallowed.
    upper = stripped.upper()
    if upper.startswith("[SILENT]"):
        return True
    return False

# ---------------------------------------------------------------------------
# Persistent thread pool for parallel cron jobs.
# The tick function submits jobs here and returns immediately so the ticker
# thread is never blocked by long-running jobs (e.g. the fixer running 15+ min).
# ---------------------------------------------------------------------------
_parallel_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_parallel_pool_max_workers: Optional[int] = None
_running_job_ids: set = set()
_running_lock = threading.Lock()

# Sequential (env-mutating) cron jobs — workdir jobs that touch
# process-global runtime state — must run one at a time, but must NOT block the
# ticker thread.  A persistent single-thread executor preserves ordering across
# ticks while keeping dispatch fire-and-forget, the same as the parallel pool.
_sequential_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


class _ReadWriteLock:
    """Writer-preferring readers-writer lock.

    Guards the process-global ``os.environ["TERMINAL_CWD"]`` override that a
    workdir cron job applies for the whole of its agent run.  Workdir jobs are
    writers: they mutate the shared env and need exclusive access.  Workdir-less
    jobs are readers: they only observe ``TERMINAL_CWD`` (indirectly, via the
    terminal / file / code-exec tools), so any number of them may run
    concurrently with each other, but none may run alongside a writer — that is
    exactly what stops a workdir-less job from picking up another job's workdir
    override and running its commands in the wrong directory.

    Writer preference bounds the wait for a workdir job (dispatched on the
    single-thread sequential pool) so a stream of workdir-less readers cannot
    starve it.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()


# Serializes the per-job TERMINAL_CWD override against every other concurrently
# running cron job.  See _ReadWriteLock and run_job for the usage contract.
_terminal_cwd_lock = _ReadWriteLock()


def _get_parallel_pool(max_workers: Optional[int]) -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent parallel pool."""
    global _parallel_pool, _parallel_pool_max_workers
    if _parallel_pool is None or _parallel_pool_max_workers != max_workers:
        if _parallel_pool is not None:
            _parallel_pool.shutdown(wait=False, cancel_futures=False)
        _parallel_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cron-parallel",
        )
        _parallel_pool_max_workers = max_workers
    return _parallel_pool


def _get_sequential_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent single-thread sequential pool.

    A single worker guarantees env-mutating jobs never overlap, even
    across ticks: a job queued by a newer tick waits for the previous tick's
    sequential jobs to finish rather than corrupting their os.environ
    state.
    """
    global _sequential_pool
    if _sequential_pool is None:
        _sequential_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cron-seq",
        )
    return _sequential_pool


def _shutdown_parallel_pool() -> None:
    """Shut down the persistent pools on process exit."""
    global _parallel_pool, _parallel_pool_max_workers, _sequential_pool
    if _parallel_pool is not None:
        _parallel_pool.shutdown(wait=True, cancel_futures=False)
        _parallel_pool = None
        _parallel_pool_max_workers = None
    if _sequential_pool is not None:
        _sequential_pool.shutdown(wait=True, cancel_futures=False)
        _sequential_pool = None


atexit.register(_shutdown_parallel_pool)


# Backward-compatible module override used by tests and emergency monkeypatches.
_hermes_home: Path | None = None


def _get_hermes_home() -> Path:
    """Resolve Hermes home dynamically while preserving test monkeypatch hooks.

    Cron is per-profile by design (#4707): the in-process ticker runs inside a
    profile-scoped gateway, so resolving the active HERMES_HOME at call time
    means a profile's jobs are stored AND executed under that profile's home
    (its .env, config.yaml, scripts, skills). Do not freeze this at import or
    anchor it at the shared default root — either re-breaks profile isolation.
    """
    return _hermes_home or get_hermes_home()


def _get_lock_paths() -> tuple[Path, Path]:
    """Resolve cron lock paths at call time so profile/env changes are honored."""
    hermes_home = _get_hermes_home()
    lock_dir = hermes_home / "cron"
    return lock_dir, lock_dir / ".tick.lock"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Treats non-dict origins (free-form provenance strings, ints, lists from
    migration scripts or hand-edited jobs.json) as missing instead of
    crashing with ``AttributeError`` on ``origin.get(...)``. Without this
    guard, a job tagged with e.g. ``"combined-digest-replaces-x-and-y"``
    crashed every fire attempt with
    ``'str' object has no attribute 'get'`` — ``mark_job_run`` recorded the
    failure, but the next tick re-loaded the same poisoned origin and
    crashed identically until the field was patched manually (#18722).
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def _cron_mirror_delivery_enabled(job: dict, cfg: Optional[dict] = None) -> bool:
    """Whether a cron delivery should also be mirrored into the target chat's
    gateway session transcript.

    Default OFF — preserves the historical isolation guarantee (cron deliveries
    live only in the cron job's own session, never the target chat's history)
    byte-for-byte for everyone who does not opt in.

    Precedence (first decisive value wins):
      1. Per-job ``attach_to_session`` (bool) — set via the ``cronjob`` tool,
         lets one briefing job opt in without flipping global behaviour.
      2. Global ``cron.mirror_delivery`` (bool) in config.yaml.
      3. False.

    When enabled, the cron's final output is appended to the target session as
    an assistant turn via the existing ``gateway.mirror.mirror_to_session`` —
    the same primitive ``send_message`` uses — so the next user reply in that
    chat sees the brief in context (no "what is Task #2?" amnesia). This is
    alternation- and cache-safe: the append lands at a turn boundary between
    user turns, never mid-loop, and never mutates the cached system prompt.
    """
    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            cfg = load_config() or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def _target_matches_origin(origin: dict, platform_name: str, chat_id: str,
                           thread_id: Optional[str]) -> bool:
    """True when a delivery target is the job's own origin conversation.

    Mirroring is scoped to the origin session by design (see
    ``_maybe_mirror_cron_delivery``). A job created from a live gateway chat
    stamps that chat as ``origin`` (``cronjob_tools._origin_from_env``), and
    that session is guaranteed to exist — it is the very conversation the user
    was in when they scheduled the job. Fan-out targets (``deliver=all``,
    explicit ``platform:chat_id`` to some *other* chat, or a home-channel
    fallback for an origin-less API/script job) are deliberately NOT mirrored:
    they are broadcasts, not a continuation of a conversation, and may point at
    a chat the user never opened an agent session in.

    This makes the historical "cold-start" worry a non-case: when the mirror
    semantically applies (target == origin) the session always exists; when no
    session exists, the target was never the origin conversation, so we simply
    do not mirror.
    """
    if not origin:
        return False
    if str(origin.get("platform", "")).lower() != str(platform_name).lower():
        return False
    if str(origin.get("chat_id", "")) != str(chat_id):
        return False
    # thread_id must match when the origin pins one (topic-scoped chats); a
    # target that lost the thread_id is not the same conversation lane.
    origin_thread = origin.get("thread_id")
    if origin_thread is not None and str(origin_thread) != str(thread_id or ""):
        return False
    return True


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (resolved once by the caller, and already scoped to
    the origin target — see ``_target_matches_origin``). Reuses the shipped
    ``mirror_to_session`` so cron rides exactly the same path that interactive
    ``send_message`` mirroring already uses, including passing ``user_id`` so a
    per-user-isolated group chat resolves to the exact member who scheduled the
    job (parity with ``send_message``). All failures are swallowed — a delivery
    that succeeded must never be reported as failed because the transcript
    mirror hit a problem.

    Because the caller only enables this for the target that equals the job's
    origin conversation, the session is expected to exist (the job was born in
    that session). A missing session therefore indicates an origin-less /
    fan-out delivery that should not have been mirrored anyway, and is treated
    as a silent no-op — never a synthetic session is created.
    """
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # Mirror as a USER turn with a labelled prefix, NOT an assistant turn.
        # The brief is not the agent speaking; an assistant-role mirror lands as
        # assistant→assistant after the agent's last turn and breaks strict
        # alternation (issue #2221, the exact failure #2313 removed). A
        # user-role turn collapses safely via repair_message_sequence's
        # consecutive-user merge on every provider, and the prefix preserves the
        # "this came from cron" context that the dropped SQLite mirror metadata
        # would otherwise lose on replay.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"), platform_name, chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"), platform_name, chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )


def _open_continuable_cron_thread(
    job: dict,
    adapter,
    chat_id: str,
    loop,
) -> Optional[str]:
    """Open a dedicated thread for a continuable cron job (thread-preferred).

    Returns the new ``thread_id`` on success, or ``None`` when the platform has
    no thread primitive (WhatsApp/Signal/SMS) or creation failed — the ``None``
    return is the caller's signal to fall back to the origin-DM mirror, the same
    open-thread-or-fallback shape as ``GatewayRunner._process_handoff``. Reuses
    the shipped ``adapter.create_handoff_thread``; no new adapter surface.
    """
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"Hermes — {task_name}"
    try:
        from agent.async_utils import safe_schedule_threadsafe

        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"), getattr(adapter, "name", "?"), e,
        )
        return None


def _seed_cron_thread_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
) -> None:
    """Seed the freshly-opened cron thread's session with the brief.

    Without this the brief is *visible* in the new thread but absent from any
    transcript, so the user's first reply in-thread would hit a session with no
    record of it ("what is Task #2?"). We create the thread-keyed session (the
    same key the user's reply will resolve to — ``build_session_key`` keys
    threads as participant-shared, so no ``user_id`` is needed) and append the
    brief as an assistant turn via the shipped ``mirror_to_session``.

    Mirrors ``GatewayRunner._process_handoff``'s seed step, but standalone:
    cron reaches the live ``SessionStore`` through the adapter's
    ``_session_store`` handle rather than the gateway object. Best-effort — a
    delivery that already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        session_store = getattr(adapter, "_session_store", None)
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=str(chat_id),
                    chat_name=chat_name,
                    chat_type="thread",
                    user_id="system:cron",
                    user_name="Cron",
                    thread_id=str(thread_id),
                )
                # Ensure the thread-keyed session row exists so the mirror has
                # a target and the user's later reply joins the same session.
                session_store.get_or_create_session(dest_source)

        from gateway.mirror import mirror_to_session

        # User-role + labelled prefix (see _maybe_mirror_cron_delivery): the
        # seeded brief must not read as an assistant turn, or the user's first
        # in-thread reply produces assistant→user→... off a phantom assistant
        # message. Pass the seed user_id so the mirror resolves the exact
        # thread-keyed session row we just created.
        mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=str(thread_id),
            user_id="system:cron",
            role="user",
        )
        logger.info(
            "Job '%s': opened continuable thread %s on %s:%s and seeded the brief",
            job.get("id", "?"), thread_id, platform_name, chat_id,
        )
    except Exception as e:
        logger.debug(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, thread_id, e,
        )


def _seed_cron_channel_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    *,
    is_dm: bool,
    user_id: Optional[str],
    chat_name: Optional[str] = None,
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` cron delivery.

    The ``in_channel`` surface (D1/D2) delivers the brief flat into the channel
    with no thread, so the continuation surface is the whole-channel /
    whole-DM session keyed ``thread_id=None`` — the same bucket
    ``reply_in_thread: false`` routes an inbound plain reply to.

    Unlike the thread path, the shipped delivery-mirror alone is NOT sufficient
    here: ``mirror_to_session`` only APPENDS to a session that already EXISTS
    (``_find_session_id`` → no-op when none matches), and a flat channel
    ``(…, None)`` row is only created when a human posts a top-level message the
    bot processes — a ``chat_postMessage`` cron delivery never goes through the
    inbound handler, so the row is usually absent and the mirror silently drops
    the brief (verified live: the brief never landed, the reply had no context).
    So we CREATE the flat session row first, exactly like
    ``_seed_cron_thread_session`` does for threads, then mirror into it.

    The session KEY must match what the user's later inbound reply resolves to
    (``build_session_key``):
    - **Channel** (``chat_type="group"``): key is
      ``…:group:<chat_id>:<user_id>`` — user-isolated — so the seed MUST carry
      the **origin's real ``user_id``** (the member who scheduled the job), NOT
      a synthetic ``system:cron`` id, or the reply keys to a different session.
    - **1:1 DM** (``chat_type="dm"``): the key is ``…:dm:<chat_id>`` and does
      NOT embed ``user_id``, so any ``user_id`` resolves to the same session.
    ``chat_type`` mirrors the inbound handler's own choice
    (``"dm" if is_dm else "group"``, ``adapter.py``), so the seeded key is
    byte-identical to the reply's key.

    Returns True if a seed row was created and the brief mirrored, else False
    (caller falls back to the plain mirror). Best-effort — a delivery that
    already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        chat_type = "dm" if is_dm else "group"
        session_store = getattr(adapter, "_session_store", None)
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=str(chat_id),
                    chat_name=chat_name,
                    chat_type=chat_type,
                    user_id=str(user_id) if user_id else None,
                    thread_id=None,  # flat — the whole-channel/DM session
                )
                # Create the flat session row so the mirror has a target and the
                # user's later plain reply joins the SAME session.
                session_store.get_or_create_session(dest_source)

        from gateway.mirror import mirror_to_session

        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=None,
            user_id=str(user_id) if user_id else None,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"), platform_name, chat_id, chat_type,
            )
        return bool(ok)
    except Exception as e:
        logger.debug(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )
        return False


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Return safe provenance details for security warnings about a cron job.

    The scheduler normally has no live HTTP request object when it detects a
    bad stored ``context_from`` reference. Including the job's saved origin
    makes future probe logs actionable without exposing secrets: platform/chat
    metadata for gateway-created jobs, and optional source-IP fields for API
    surfaces that persist them in origin metadata.
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""

    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""


def _plugin_cron_env_var(platform_name: str) -> str:
    """Return the cron home-channel env var registered by a plugin platform.

    Falls through the platform registry so plugins that set
    ``cron_deliver_env_var`` on their ``PlatformEntry`` get cron delivery
    support without editing this module.
    """
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name.lower())
        if entry and entry.cron_deliver_env_var:
            return entry.cron_deliver_env_var
    except Exception:
        pass
    return ""


def _is_known_delivery_platform(platform_name: str) -> bool:
    """Whether ``platform_name`` is a valid cron delivery target.

    Hardcoded built-ins in ``_KNOWN_DELIVERY_PLATFORMS`` are checked first;
    plugin platforms registered via ``PlatformEntry`` are accepted if they
    provide a ``cron_deliver_env_var``.
    """
    name = platform_name.lower()
    if name in _KNOWN_DELIVERY_PLATFORMS:
        return True
    return bool(_plugin_cron_env_var(name))


def _resolve_home_env_var(platform_name: str) -> str:
    """Return the env var name for a platform's cron home channel.

    Built-in platforms are in ``_HOME_TARGET_ENV_VARS``; plugin platforms are
    resolved from the platform registry.
    """
    name = platform_name.lower()
    env_var = _HOME_TARGET_ENV_VARS.get(name)
    if env_var:
        return env_var
    return _plugin_cron_env_var(name)


def _get_home_target_chat_id(platform_name: str) -> str:
    """Return the configured home target chat/room ID for a delivery platform."""
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return ""
    value = os.getenv(env_var, "")
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(legacy, "")
    return value


def _get_home_target_thread_id(platform_name: str) -> Optional[str]:
    """Return the optional thread/topic ID for a platform home target.

    Telegram-only override: ``TELEGRAM_CRON_THREAD_ID`` takes precedence over
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID`` for cron delivery. When topic mode is
    enabled, deliveries that land in the root DM (thread_id unset) end up in
    the system-only lobby where the user cannot reply — the gateway returns
    the lobby reminder and drops ``reply_to_message_id`` (#24409). Pointing
    cron at a dedicated topic via this env var lets replies work as expected
    without changing the lobby invariant.
    """
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return None
    if platform_name.lower() == "telegram":
        cron_thread = os.getenv("TELEGRAM_CRON_THREAD_ID", "").strip()
        if cron_thread:
            return cron_thread
    value = os.getenv(f"{env_var}_THREAD_ID", "").strip()
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(f"{legacy}_THREAD_ID", "").strip()
    return value or None


def _iter_home_target_platforms():
    """Iterate built-in + plugin platform names that expose a home channel.

    Used by the ``deliver=origin`` fallback when the job has no origin.
    """
    for name in _HOME_TARGET_ENV_VARS:
        yield name
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.cron_deliver_env_var and entry.name not in _HOME_TARGET_ENV_VARS:
                yield entry.name
    except Exception:
        pass


def cron_delivery_targets() -> list[dict]:
    """Return the platforms a cron job can auto-deliver to.

    Single source of truth for any UI (dashboard dropdown, etc.) that lets a
    user pick a cron delivery target. A platform is included when it is a valid
    cron delivery platform AND its gateway is configured (enabled + credentials
    present). Each entry reports whether the platform's home target (the
    room/channel cron posts to) is set — a platform can be configured for
    interactive use but still lack the home target an unattended cron job needs.

    Returns a list of dicts: ``{"id", "name", "home_target_set", "home_env_var"}``
    ordered by the gateway's canonical platform order. Callers should always
    prepend the implicit ``local`` option themselves — it needs no config.
    """
    targets: list[dict] = []
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        connected = {p.value for p in gateway_config.get_connected_platforms()}
    except Exception:
        logger.debug("cron_delivery_targets: gateway config unavailable", exc_info=True)
        connected = set()

    for name in _iter_home_target_platforms():
        if name not in connected:
            continue
        if not _is_known_delivery_platform(name):
            continue
        env_var = _resolve_home_env_var(name)
        targets.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "home_target_set": bool(_get_home_target_chat_id(name)),
                "home_env_var": env_var or None,
            }
        )
    return targets


def _resolve_single_delivery_target(job: dict, deliver_value: str) -> Optional[dict]:
    """Resolve one concrete auto-delivery target for a cron job."""

    origin = _resolve_origin(job)

    if deliver_value == "local":
        return None

    if deliver_value == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": origin.get("thread_id"),
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in _iter_home_target_platforms():
            chat_id = _get_home_target_chat_id(platform_name)
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": _get_home_target_thread_id(platform_name),
                }
        return None

    if ":" in deliver_value:
        platform_name, rest = deliver_value.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import _parse_target_ref

        parsed_chat_id, parsed_thread_id, is_explicit = _parse_target_ref(platform_key, rest)
        if is_explicit:
            chat_id, thread_id = parsed_chat_id, parsed_thread_id
        else:
            chat_id, thread_id = rest, None

        # Resolve human-friendly labels like "Alice (dm)" to real IDs.
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_key, chat_id)
            if resolved:
                parsed_chat_id, parsed_thread_id, resolved_is_explicit = _parse_target_ref(platform_key, resolved)
                if resolved_is_explicit:
                    chat_id = parsed_chat_id
                    if parsed_thread_id is not None:
                        thread_id = parsed_thread_id
                else:
                    chat_id = resolved
        except Exception:
            pass

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

    platform_name = deliver_value
    if origin and origin.get("platform") == platform_name:
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if not _is_known_delivery_platform(platform_name):
        return None
    chat_id = _get_home_target_chat_id(platform_name)
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": _get_home_target_thread_id(platform_name),
    }


def _normalize_deliver_value(deliver) -> str:
    """Normalize a stored/submitted ``deliver`` value to its canonical string form.

    The contract is that ``deliver`` is a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:-1001:17"``, or comma-separated combinations).
    Historically some callers — MCP clients passing an array, direct edits of
    ``jobs.json``, or stale code paths — have stored a list/tuple like
    ``["telegram"]``.  ``str(["telegram"])`` would serialize to the literal
    string ``"['telegram']"``, which is not a known platform and fails
    resolution silently.  Flatten lists/tuples into a comma-separated string
    so both forms work.  Returns ``"local"`` for anything falsy.
    """
    if deliver is None or deliver == "":
        return "local"
    if isinstance(deliver, (list, tuple)):
        parts = [str(p).strip() for p in deliver if str(p).strip()]
        return ",".join(parts) if parts else "local"
    return str(deliver)


# Routing intent tokens — resolved at fire time, not create time, so a
# job created before Telegram was wired up will pick up Telegram once it
# comes online.  ``all`` expands into the set of connected platforms
# (those with a configured home chat_id) in _expand_routing_tokens.
_ROUTING_TOKENS = frozenset({"all"})


def _expand_routing_tokens(part: str) -> List[str]:
    """Expand a routing-intent token to concrete platform names.

    ``all`` expands to every platform in ``_iter_home_target_platforms()``
    that has a configured home chat_id right now.  Unknown / non-token
    values pass through unchanged as a single-element list, so the caller
    can treat every token uniformly.
    """
    token = part.lower()
    if token not in _ROUTING_TOKENS:
        return [part]
    expanded: List[str] = []
    for platform_name in _iter_home_target_platforms():
        if _get_home_target_chat_id(platform_name):
            expanded.append(platform_name)
    return expanded


def _resolve_delivery_targets(job: dict) -> List[dict]:
    """Resolve all concrete auto-delivery targets for a cron job.

    Accepts the legacy comma-separated ``deliver`` string plus the
    ``all`` routing-intent token, which expands to every platform with
    a configured home channel.  Tokens may be combined with explicit
    targets: ``origin,all`` and ``all,telegram:-100:17`` both work.
    Duplicate (platform, chat_id, thread_id) tuples are collapsed by the
    existing dedup pass.
    """
    deliver = _normalize_deliver_value(job.get("deliver", "local"))
    if deliver == "local":
        return []

    raw_parts = [p.strip() for p in deliver.split(",") if p.strip()]

    # Expand routing intents.
    parts: List[str] = []
    for raw in raw_parts:
        parts.extend(_expand_routing_tokens(raw))

    seen = set()
    targets = []
    for part in parts:
        target = _resolve_single_delivery_target(job, part)
        if target:
            key = (target["platform"].lower(), str(target["chat_id"]), target.get("thread_id"))
            if key not in seen:
                seen.add(key)
                targets.append(target)
    return targets


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    targets = _resolve_delivery_targets(job)
    return targets[0] if targets else None


# Media extension sets — audio routing is centralized in gateway.platforms.base
# via should_send_media_as_audio() so Telegram-specific rules stay in one place.
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter,
    chat_id: str,
    media_files: list,
    metadata: dict | None,
    loop,
    job: dict,
    platform=None,
) -> None:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.
    """
    from pathlib import Path

    from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)

    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            route_platform = platform if platform is not None else getattr(adapter, "platform", None)
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                logger.warning(
                    "Job '%s': cannot send media %s, gateway loop unavailable",
                    job.get("id", "?"), media_path,
                )
                return
            try:
                result = future.result(timeout=30)
            except TimeoutError:
                future.cancel()
                raise
            if result and not getattr(result, "success", True):
                logger.warning(
                    "Job '%s': media send failed for %s: %s",
                    job.get("id", "?"), media_path, getattr(result, "error", "unknown"),
                )
        except Exception as e:
            logger.warning("Job '%s': failed to send media %s: %s", job.get("id", "?"), media_path, e)


def _confirm_adapter_delivery(send_result) -> bool:
    """Return True only if ``send_result`` unambiguously confirms delivery.

    A live adapter that returns ``None`` (e.g. a swallowed exception, a busy
    platform, or a code path that returns early without producing a
    ``SendResult``) must NOT be treated as success — doing so causes the
    scheduler to log ``"delivered to <chat> via live adapter"`` while the
    gateway never actually sees the message (#47056).

    Likewise, an object missing a ``success`` attribute (e.g. a bare ``dict``
    or a partial mock) is a contract violation: it does not actually tell us
    whether the send succeeded.  Require an explicit, truthy ``success``
    attribute to count as confirmed.
    """
    if send_result is None:
        return False
    if not hasattr(send_result, "success"):
        return False
    return bool(getattr(send_result, "success"))


def _deliver_result(job: dict, content: str, adapters=None, loop=None) -> Optional[str]:
    """
    Deliver job output to the configured target(s) (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    Returns None on success, or an error string on failure.
    """
    targets = _resolve_delivery_targets(job)
    if not targets:
        deliver_value = _normalize_deliver_value(job.get("deliver", "local"))
        if deliver_value == "local":
            return None  # local-only jobs don't deliver — not a failure
        # deliver=origin with no resolvable origin and no configured home
        # channels: treat as local rather than reporting an error.  CLI-created
        # jobs never capture a {platform, chat_id} origin, so failing here would
        # make every CLI `deliver=origin` (or auto-detect) job emit a spurious
        # "no delivery target resolved" error on every run (#43014).  The output
        # is still persisted in last_output for `cron list`/resume.
        if deliver_value == "origin":
            logger.info(
                "Job '%s': deliver=origin but no origin or home channels — "
                "skipping delivery (output saved in last_output)",
                job.get("name", job.get("id", "?")),
            )
            return None
        msg = f"no delivery target resolved for deliver={deliver_value}"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    user_cfg = None
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        job_id = job.get("id", "")
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"(job_id: {job_id})\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"To stop or manage this job, send me a new message (e.g. \"stop reminder {task_name}\")."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import BasePlatformAdapter
    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)

    # Resolve the delivery-mirror gate ONCE (default off). When on, each
    # successful delivery is also appended to the target chat's gateway session
    # transcript so a user reply in that chat sees the cron output in context.
    # Mirror the CLEAN, unwrapped output (not the cron header/footer).
    try:
        mirror_enabled = _cron_mirror_delivery_enabled(job, user_cfg)
    except Exception:
        mirror_enabled = False
    mirror_text = ""
    if mirror_enabled:
        _, mirror_text = BasePlatformAdapter.extract_media(content)
        mirror_text = (mirror_text or "").strip()

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    delivery_errors = []

    for target in targets:
        platform_name = target["platform"]
        chat_id = target["chat_id"]
        thread_id = target.get("thread_id")

        # Diagnostic: log thread_id for topic-aware delivery debugging
        origin = _resolve_origin(job) or {}
        origin_thread = origin.get("thread_id")
        if origin_thread and not thread_id:
            logger.warning(
                "Job '%s': origin has thread_id=%s but delivery target lost it "
                "(deliver=%s, target=%s)",
                job["id"], origin_thread, job.get("deliver", "local"), target,
            )
        elif thread_id:
            logger.debug(
                "Job '%s': delivering to %s:%s thread_id=%s",
                job["id"], platform_name, chat_id, thread_id,
            )

        # Mirror is scoped to the ORIGIN conversation only. A fan-out / broadcast
        # / home-channel-fallback target is never mirrored (it is not the
        # conversation the job was created in, and may have no session at all).
        mirror_this_target = mirror_enabled and _target_matches_origin(
            origin, platform_name, chat_id, thread_id
        )
        # Pass the origin's user_id so a per-user-isolated group chat resolves to
        # the exact member who scheduled the job — parity with send_message.
        origin_user_id = origin.get("user_id") if mirror_this_target else None

        # Built-in names resolve to their enum member; plugin platform names
        # create dynamic members via Platform._missing_().
        try:
            platform = Platform(platform_name.lower())
        except (ValueError, KeyError):
            msg = f"unknown platform '{platform_name}'"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        pconfig = config.platforms.get(platform)
        if not pconfig or not pconfig.enabled:
            msg = f"platform '{platform_name}' not configured/enabled"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        # Prefer the live adapter when the gateway is running — this supports E2EE
        # rooms (e.g. Matrix) where the standalone HTTP path cannot encrypt.
        runtime_adapter = (adapters or {}).get(platform)
        delivered = False
        target_errors = []

        # Continuable cron surface (D1/D2/D6): resolve the delivery surface for
        # this platform generically from its config ``extra``. Default "thread"
        # (today's behaviour, byte-identical). "in_channel" delivers the brief
        # FLAT into the channel (no dedicated thread) so a plain channel reply
        # continues the job in-context via the shared-channel session
        # ``(platform, chat_id, None)`` — the same bucket ``reply_in_thread:
        # false`` routes inbound channel messages to. The key is read
        # generically here (any platform); the ``in_channel`` branch is gated on
        # the adapter capability flag ``supports_inchannel_continuable`` so an
        # unsupported platform fails SAFE to "thread" (Slack is the first
        # consumer; "first consumer ≠ definition").
        surface_mode = "thread"
        try:
            surface_raw = (pconfig.extra or {}).get("cron_continuable_surface")
            if surface_raw is not None and str(surface_raw).strip().lower() == "in_channel":
                surface_mode = "in_channel"
        except Exception:
            surface_mode = "thread"
        in_channel_surface = surface_mode == "in_channel"
        if in_channel_surface and runtime_adapter is not None and not getattr(
            runtime_adapter, "supports_inchannel_continuable", False
        ):
            # Fail safe (D6): platform has no in_channel continuation primitive.
            logger.debug(
                "Job '%s': cron_continuable_surface=in_channel not supported on "
                "%s, using thread",
                job.get("id", "?"), platform_name,
            )
            in_channel_surface = False

        # For an in_channel delivery the flat continuation session is created
        # explicitly below (the shipped mirror only APPENDS to an existing
        # session, and the flat channel row is otherwise absent for a
        # chat_postMessage delivery). ``is_dm`` selects the session chat_type so
        # the seeded key matches the inbound reply's key: a 1:1 DM keys as
        # ``dm`` (Slack DM channel ids start with "D"; or the origin says so),
        # everything else as ``group`` (shared channel). ``inchannel_seeded``
        # suppresses the generic mirror below so the brief is not double-written.
        origin_chat_type = str(origin.get("chat_type") or "").lower()
        is_dm_target = origin_chat_type == "dm" or (
            not origin_chat_type and str(chat_id).startswith("D")
        )
        inchannel_seeded = False

        # Continuable cron (thread-preferred): when mirroring is enabled for the
        # origin target and the gateway is live, try to open a DEDICATED thread
        # for this job and deliver the brief into it. On thread-capable
        # platforms (Telegram/Discord/Slack) the brief + the user's replies live
        # in their own scrollback; the thread-keyed session is seeded so a reply
        # continues with full context. On DM-only platforms (WhatsApp/Signal)
        # create_handoff_thread returns None and we fall back to mirroring into
        # the origin DM session (handled after delivery). Cf. _process_handoff.
        #
        # in_channel surface (D2): SKIP thread creation entirely — leave
        # thread_id=None so the delivery posts flat, then
        # ``_seed_cron_channel_session`` (below) CREATES the shared-channel
        # session and mirrors the brief into it. The shipped mirror alone is
        # NOT enough here: ``mirror_to_session`` only APPENDS to an existing
        # session and a flat ``(platform, chat_id, None)`` row is otherwise
        # absent for a ``chat_postMessage`` delivery, so the seed must create
        # the row first (F5).
        thread_seeded = False
        opened_thread_id: Optional[str] = None
        if (
            mirror_this_target
            and not in_channel_surface
            and runtime_adapter is not None
            and loop is not None
            and not thread_id  # never override an explicit origin thread/topic
        ):
            new_thread_id = _open_continuable_cron_thread(
                job, runtime_adapter, chat_id, loop,
            )
            if new_thread_id:
                # Route THIS delivery into the new thread now (the send needs the
                # thread_id), but defer seeding the thread session until the
                # delivery actually succeeds — otherwise an open-succeeds /
                # deliver-fails case leaves a seeded brief the user never saw,
                # and (worse) suppresses the DM-fallback mirror via thread_seeded.
                thread_id = new_thread_id
                opened_thread_id = new_thread_id

        if runtime_adapter is not None and loop is not None and getattr(loop, "is_running", lambda: False)():
            # Telegram three-mode topic routing (#22773): a private chat
            # (positive chat_id) with a NUMERIC topic id is a Bot API Direct
            # Messages topic and must be addressed via ``direct_messages_topic_id``
            # — a bare ``message_thread_id`` is rejected/mis-routed by Bot API
            # 10.0 and lands in General.  Forum/supergroup targets (negative
            # chat_id) and named DM-topic lanes keep the default thread_id
            # handling.  Compute the routed metadata ONCE so both the text send
            # (via DeliveryRouter) and the media send use the same routing.
            from gateway.delivery import (
                DeliveryRouter,
                DeliveryTarget,
                _looks_like_int,
                looks_like_telegram_private_chat_id,
            )

            is_private_dm_topic = (
                platform == Platform.TELEGRAM
                and thread_id is not None
                and looks_like_telegram_private_chat_id(str(chat_id))
                and _looks_like_int(str(thread_id))
            )
            if is_private_dm_topic:
                # Routed via direct_messages_topic_id (mode 2), no bare thread_id.
                route_thread_id = None
                route_metadata = {
                    "direct_messages_topic_id": str(thread_id),
                    "job_id": job["id"],
                }
                # Media metadata mirrors the text routing so attachments land in
                # the same DM topic instead of the General lane (#22773).
                media_metadata = {"direct_messages_topic_id": str(thread_id)}
            else:
                route_thread_id = str(thread_id) if thread_id is not None else None
                route_metadata = {"job_id": job["id"]}
                media_metadata = {"thread_id": thread_id} if thread_id else None

            try:
                # Send cleaned text (MEDIA tags stripped) — not the raw content.
                # Route through the gateway's DeliveryRouter so the live send
                # gets the same platform-specific routing as live messages —
                # in particular Telegram's three-mode topic routing.  The
                # standalone cron path lacked this, so DM-topic cron deliveries
                # landed in the General topic or were rejected by Bot API 10.0
                # (#22773).
                text_to_send = cleaned_delivery_content.strip()
                adapter_ok = True
                timed_out = False
                if text_to_send:
                    from agent.async_utils import safe_schedule_threadsafe

                    router = DeliveryRouter(config, adapters)
                    route_target = DeliveryTarget(
                        platform=platform,
                        chat_id=str(chat_id),
                        thread_id=route_thread_id,
                        is_explicit=True,
                    )
                    # Pass thread routing via the target (not a bare metadata
                    # "thread_id"): the router only applies its Telegram DM-topic
                    # detection when "thread_id"/"message_thread_id" are absent
                    # from metadata, deriving the routing from target.thread_id
                    # or the explicit direct_messages_topic_id above.
                    future = safe_schedule_threadsafe(
                        router._deliver_to_platform(
                            route_target,
                            text_to_send,
                            route_metadata,
                        ),
                        loop,
                    )
                    if future is None:
                        adapter_ok = False
                        target_errors.append("live adapter event loop scheduling failed")
                    else:
                        send_result = None
                        timeout_handled = False
                        try:
                            send_result = future.result(timeout=60)
                        except TimeoutError:
                            # #38922: a slow confirmation does NOT necessarily
                            # mean the send failed — but we must distinguish two
                            # cases via future.cancel()'s return value:
                            #
                            #   cancel() == False -> the coroutine was already
                            #     running on the gateway loop when the timeout
                            #     fired; the request is in flight on the wire and
                            #     cannot be un-sent.  Re-sending via standalone
                            #     would be a guaranteed DUPLICATE, so treat it as
                            #     delivered (assume-delivered).
                            #
                            #   cancel() == True -> the scheduled callback never
                            #     started executing (loop wedged/backlogged for
                            #     the full 60s), so nothing was sent.  We MUST
                            #     fall through to the standalone path or the
                            #     message is silently dropped (worse than a
                            #     duplicate).
                            cancelled = future.cancel()
                            if cancelled:
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    "timed out before the coroutine was dispatched"
                                )
                                logger.warning(
                                    "Job '%s': %s, falling back to standalone",
                                    job["id"], msg,
                                )
                                target_errors.append(msg)
                                adapter_ok = False  # fall through to standalone path
                                timeout_handled = True
                            else:
                                timed_out = True
                                timeout_handled = True
                                logger.warning(
                                    "Job '%s': live adapter send to %s:%s timed out "
                                    "after 60s; already dispatched (in flight), "
                                    "assuming delivered (skipping standalone fallback "
                                    "to avoid duplicate)",
                                    job["id"], platform_name, chat_id,
                                )
                        except Exception as ex:
                            # A real send error (not a slow confirmation) — fall
                            # through to the standalone path so the message is
                            # still delivered.
                            target_errors.append(f"live adapter send failed: {ex}")
                            raise

                        if timeout_handled:
                            # The timeout branch above already decided the
                            # outcome (assume-delivered if in flight, or
                            # adapter_ok=False to fall through if never
                            # dispatched).  send_result is None, so skip the
                            # confirmation/thread-fallback inspection below.
                            pass
                        else:
                            # _deliver_to_platform returns either a SendResult
                            # (.success attr) or, when the silence-narration
                            # filter drops the message, a plain dict
                            # {"success": True, "delivered": False, ...}.
                            # Normalize both shapes so a getattr default doesn't
                            # misread a dict, and so a None / success-less object
                            # is NOT counted as delivered (#47056).
                            if isinstance(send_result, dict):
                                send_success = bool(send_result.get("success", False))
                                send_raw_response = send_result.get("raw_response")
                            else:
                                send_success = _confirm_adapter_delivery(send_result)
                                send_raw_response = getattr(send_result, "raw_response", None)

                            if not send_success:
                                if isinstance(send_result, dict):
                                    err = send_result.get("error", "unknown")
                                    shape = "dict"
                                elif send_result is not None:
                                    err = getattr(send_result, "error", None)
                                    shape = type(send_result).__name__
                                else:
                                    err = "no response from adapter"
                                    shape = "None"
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    f"returned unconfirmed result ({shape}, error={err})"
                                )
                                logger.warning(
                                    "Job '%s': %s, falling back to standalone",
                                    job["id"], msg,
                                )
                                target_errors.append(msg)
                                adapter_ok = False  # fall through to standalone path
                            elif (
                                send_raw_response
                                and thread_id
                                and send_raw_response.get("thread_fallback")
                            ):
                                requested_thread_id = send_raw_response.get("requested_thread_id") or thread_id
                                msg = (
                                    f"configured thread_id {requested_thread_id} for "
                                    f"{platform_name}:{chat_id} was not found; delivered without thread_id"
                                )
                                logger.warning("Job '%s': %s", job["id"], msg)
                                delivery_errors.append(msg)

                # Send extracted media files as native attachments via the live
                # adapter, using the same DM-topic-aware routing as the text send
                # (#22773 — media previously used a bare thread_id and landed in
                # the General lane for private DM topics).  Skip on an in-flight
                # confirmation timeout: the gateway loop is contended, so each
                # media send would also block its 30s budget, and the text
                # payload is already assumed delivered (#38922).  Record the
                # skipped attachments so the drop is visible rather than silently
                # lost.
                if adapter_ok and not timed_out and media_files:
                    _send_media_via_adapter(
                        runtime_adapter,
                        chat_id,
                        media_files,
                        media_metadata,
                        loop,
                        job,
                        platform=platform,
                    )
                elif timed_out and media_files:
                    msg = (
                        f"{len(media_files)} media attachment(s) not delivered to "
                        f"{platform_name}:{chat_id} (live adapter confirmation timed out)"
                    )
                    logger.warning("Job '%s': %s", job["id"], msg)
                    delivery_errors.append(msg)

                if adapter_ok:
                    logger.info("Job '%s': delivered to %s:%s via live adapter", job["id"], platform_name, chat_id)
                    delivered = True
                    # Seed the thread session only now that delivery into it
                    # succeeded (deferred from thread-open above).
                    if opened_thread_id and not thread_seeded:
                        _seed_cron_thread_session(
                            job, runtime_adapter, platform_name, chat_id,
                            opened_thread_id, mirror_text,
                            chat_name=origin.get("chat_name"),
                        )
                        thread_seeded = True
                    # in_channel surface: CREATE + seed the flat channel/DM
                    # session (the shipped mirror only appends to an existing
                    # session — the flat row is otherwise absent for a
                    # chat_postMessage delivery, so the brief would be lost).
                    if in_channel_surface and mirror_this_target and not thread_seeded:
                        inchannel_seeded = _seed_cron_channel_session(
                            job, runtime_adapter, platform_name, chat_id,
                            mirror_text, is_dm=is_dm_target,
                            user_id=origin_user_id,
                            chat_name=origin.get("chat_name"),
                        )
                    _maybe_mirror_cron_delivery(
                        job, platform_name, chat_id, mirror_text,
                        thread_id=thread_id, user_id=origin_user_id,
                        enabled=mirror_this_target and not thread_seeded and not inchannel_seeded,
                    )
            except Exception as e:
                err_msg = f"live adapter delivery to {platform_name}:{chat_id} failed: {e}"
                if not any(err_msg in err for err in target_errors):
                    target_errors.append(err_msg)
                logger.warning(
                    "Job '%s': %s, falling back to standalone",
                    job["id"], err_msg,
                )

        if not delivered:
            # Standalone path: run the async send in a fresh event loop (safe from any thread)
            coro = _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files)
            try:
                result = asyncio.run(coro)
            except RuntimeError:
                # asyncio.run() checks for a running loop before awaiting the coroutine;
                # when it raises, the original coro was never started — close it to
                # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
                # fresh thread that has no running loop.
                coro.close()
                # The thread-pool fallback can itself raise (SMTP ConnectionError,
                # future.result timeout, etc.). An exception raised inside this
                # `except RuntimeError` block is NOT caught by the sibling
                # `except Exception` below — it would escape _deliver_result()
                # and crash the whole delivery loop, silently skipping every
                # remaining target (#47163). Wrap the fallback in its own
                # try/except so a per-target failure is logged and the loop
                # continues to the next target.
                try:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files))
                        result = future.result(timeout=30)
                    finally:
                        pool.shutdown(wait=False)
                except Exception as e:
                    msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                    logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                    target_errors.extend([msg])
                    delivery_errors.extend(target_errors)
                    continue
            except Exception as e:
                msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            if result and result.get("error"):
                msg = f"delivery error: {result['error']}"
                logger.error("Job '%s': %s", job["id"], msg)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
            _maybe_mirror_cron_delivery(
                job, platform_name, chat_id, mirror_text,
                thread_id=thread_id, user_id=origin_user_id,
                enabled=mirror_this_target and not thread_seeded,
            )

    if delivery_errors:
        return "; ".join(delivery_errors)
    return None


_DEFAULT_SCRIPT_TIMEOUT = 3600  # seconds (1 hour)
# Backward-compatible module override used by tests and emergency monkeypatches.
_SCRIPT_TIMEOUT = _DEFAULT_SCRIPT_TIMEOUT


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _SCRIPT_TIMEOUT != _DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = int(float(_SCRIPT_TIMEOUT))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", _SCRIPT_TIMEOUT)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return _DEFAULT_SCRIPT_TIMEOUT


def _run_job_script(script_path: str) -> tuple[bool, str]:
    """Execute a cron job's data-collection script and capture its output.

    Scripts must reside within HERMES_HOME/scripts/.  Both relative and
    absolute paths are resolved and validated against this directory to
    prevent arbitrary script execution via path traversal or absolute
    path injection.

    Supported interpreters (chosen by file extension):

    * ``.sh`` / ``.bash`` — run with ``/bin/bash``
    * anything else — run with the current Python interpreter
      (``sys.executable``), preserving the original behaviour for
      Python-based pre-check and data-collection scripts.

    Shell support lets ``no_agent=True`` jobs ship classic bash watchdogs
    (the `memory-watchdog.sh` pattern) without wrapping them in Python.

    Subprocess environment is passed through ``_sanitize_subprocess_env`` so
    provider credentials and other Hermes-managed secrets are not inherited
    (SECURITY.md §2.3), matching terminal and MCP child processes.

    Args:
        script_path: Path to the script.  Relative paths are resolved
            against HERMES_HOME/scripts/.  Absolute and ~-prefixed paths
            are also validated to ensure they stay within the scripts dir.

    Returns:
        (success, output) — on failure *output* contains the error message so the
        LLM can report the problem to the user.
    """
    scripts_dir = _get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    # Guard against path traversal, absolute path injection, and symlink
    # escape — scripts MUST reside within HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = _get_script_timeout()

    # Pick an interpreter by extension.  Bash for .sh/.bash, Python for
    # everything else.  We deliberately do NOT honour the file's own
    # shebang: the scripts dir is trusted, but keeping the interpreter
    # choice explicit here keeps the allowed surface small and auditable.
    suffix = path.suffix.lower()
    if suffix in {".sh", ".bash"}:
        # Resolve bash dynamically so Windows (Git Bash) and Linux/macOS
        # all work.  On native Windows without Git for Windows installed
        # shutil.which returns None — fall back to a clear error rather
        # than a FileNotFoundError with a confusing "[WinError 2]"
        # traceback.
        _bash = shutil.which("bash") or (
            "/bin/bash" if os.path.isfile("/bin/bash") else None
        )
        if _bash is None:
            return False, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
            )
        argv = [_bash, str(path)]
    else:
        argv = [sys.executable, str(path)]

    try:
        from tools.environments.local import _sanitize_subprocess_env

        popen_kwargs = {"creationflags": windows_hide_flags()} if sys.platform == "win32" else {}
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=script_timeout,
            cwd=str(path.parent),
            env=_sanitize_subprocess_env(os.environ.copy()),
            **popen_kwargs,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        # Redact secrets from both stdout and stderr before any return path.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as e:
            logger.warning("Failed to redact sensitive text from output: %s", e)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {script_timeout}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _parse_wake_gate(script_output: str) -> bool:
    """Parse the last non-empty stdout line of a cron job's pre-check script
    as a wake gate.

    The convention (ported from nanoclaw #1232): if the last stdout line is
    JSON like ``{"wakeAgent": false}``, the agent is skipped entirely — no
    LLM run, no delivery. Any other output (non-JSON, missing flag, gate
    absent, or ``wakeAgent: true``) means wake the agent normally.

    Returns True if the agent should wake, False to skip.
    """
    if not script_output:
        return True
    stripped_lines = [line for line in script_output.splitlines() if line.strip()]
    if not stripped_lines:
        return True
    last_line = stripped_lines[-1].strip()
    try:
        gate = json.loads(last_line)
    except (json.JSONDecodeError, ValueError):
        return True
    if not isinstance(gate, dict):
        return True
    return gate.get("wakeAgent", True) is not False


def _build_job_prompt(job: dict, prerun_script: Optional[tuple] = None) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first.

    Args:
        job: The cron job dict.
        prerun_script: Optional ``(success, stdout)`` from a script that has
            already been executed by the caller (e.g. for a wake-gate check).
            When provided, the script is not re-executed and the cached
            result is used for prompt injection. When omitted, the script
            (if any) runs inline as before.
    """
    user_prompt = str(job.get("prompt") or "")
    prompt = user_prompt
    skills = job.get("skills")
    # True when runtime-collected DATA (script stdout, upstream-job output)
    # has been injected into the prompt. Data content legitimately quotes
    # command-shape strings (a triage feed ingesting a bug report that
    # pastes `rm -rf /`), so it must not be scanned with the strict
    # user-prompt pattern set — see _scan_assembled_cron_prompt.
    has_injected_data = False

    # Run data-collection script if configured, inject output as context.
    script_path = job.get("script")
    if script_path:
        if prerun_script is not None:
            success, script_output = prerun_script
        else:
            success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
                has_injected_data = True
            else:
                # Script produced no output — nothing to report, skip AI call.
                return None
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )
            has_injected_data = True

    # Inject output from referenced cron jobs as context.
    context_from = job.get("context_from")
    if context_from:
        from cron.jobs import OUTPUT_DIR
        if isinstance(context_from, str):
            context_from = [context_from]
        for source_job_id in context_from:
            # Guard against path traversal — valid job IDs are 12-char hex strings
            if not source_job_id or not all(c in "0123456789abcdef" for c in source_job_id):
                logger.warning(
                    "context_from: skipping invalid job_id %r for job_id=%r name=%r%s",
                    source_job_id,
                    job.get("id"),
                    job.get("name"),
                    _cron_job_origin_log_suffix(job),
                )
                continue
            try:
                job_output_dir = OUTPUT_DIR / source_job_id
                if not job_output_dir.exists():
                    continue  # silent skip — no output yet
                output_files = sorted(
                    job_output_dir.glob("*.md"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not output_files:
                    continue  # silent skip — no output yet
                latest_output = output_files[0].read_text(encoding="utf-8").strip()
                # Truncate to 8K characters to avoid prompt bloat
                _MAX_CONTEXT_CHARS = 8000
                if len(latest_output) > _MAX_CONTEXT_CHARS:
                    latest_output = latest_output[:_MAX_CONTEXT_CHARS] + "\n\n[... output truncated ...]"
                if latest_output:
                    prompt = (
                        f"## Output from job '{source_job_id}'\n"
                        "The following is the most recent output from a preceding "
                        "cron job. Use it as context for your analysis.\n\n"
                        f"```\n{latest_output}\n```\n\n"
                        f"{prompt}"
                    )
                    has_injected_data = True
                else:
                    continue  # silent skip — empty output
            except (OSError, PermissionError) as e:
                logger.warning("context_from: failed to read output for job %r: %s", source_job_id, e)
                # silent skip — do not pollute the prompt with error messages

    # Always prepend cron execution guidance so the agent knows how
    # delivery works and can suppress delivery when appropriate.
    cron_hint = (
        "[IMPORTANT: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return _scan_assembled_cron_prompt(
            prompt,
            job,
            has_skills=False,
            has_injected_data=has_injected_data,
            user_prompt=user_prompt,
        )

    from tools.skills_tool import skill_view
    from tools.skill_usage import bump_use
    from agent.skill_bundles import build_bundle_invocation_message, resolve_bundle_command_key

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        # Cron jobs historically accepted only skill names here, but the CLI/gateway
        # slash-command path lets bundles shadow skills with the same slug. Mirror
        # that behavior so `skills: ["my-bundle"]` expands bundle members instead
        # of being treated as a missing skill.
        bundle_key = resolve_bundle_command_key(skill_name.lstrip("/"))
        if bundle_key:
            bundle_payload = build_bundle_invocation_message(
                bundle_key,
                user_instruction="",
                task_id=str(job.get("id") or "") or None,
            )
            if bundle_payload:
                bundle_message, _loaded_bundle_skills, _missing_bundle_skills = bundle_payload
                if parts:
                    parts.append("")
                parts.append(bundle_message)
                continue
            logger.warning(
                "Cron job '%s': bundle '%s' could not load any skills, skipping",
                job.get("name", job.get("id")),
                skill_name,
            )
            skipped.append(skill_name)
            continue

        try:
            loaded = json.loads(skill_view(skill_name))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Cron job '%s': skill '%s' returned invalid JSON, skipping", job.get("name", job.get("id")), skill_name)
            skipped.append(skill_name)
            continue
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        # Bump usage so the curator sees this skill as actively used.
        try:
            bump_use(skill_name)
        except Exception:
            logger.debug("Cron job: failed to bump skill usage for '%s'", skill_name, exc_info=True)

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[IMPORTANT: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])
    return _scan_assembled_cron_prompt("\n".join(parts), job, has_skills=True)


def _scan_assembled_cron_prompt(
    assembled: str,
    job: dict,
    *,
    has_skills: bool = False,
    has_injected_data: bool = False,
    user_prompt: Optional[str] = None,
) -> str:
    """Scan the fully-assembled cron prompt for injection patterns. Raises
    ``CronPromptInjectionBlocked`` when a match fires so ``run_job`` can
    surface a clear refusal to the operator.

    Plugs the #3968 gap: ``_scan_cron_prompt`` runs on the user-supplied
    prompt at create/update, but skill content is loaded from disk at
    runtime and was never scanned. Since cron runs non-interactively
    (auto-approves tool calls), a malicious skill carrying an injection
    payload bypassed every gate.

    Two pattern tiers, selected by what the assembled prompt CONTAINS,
    not just whether skills are attached:

    - When the assembled prompt is essentially the user prompt + the cron
      hint (no skills, no injected data), the STRICT ``_scan_cron_prompt``
      patterns apply: a bare ``rm -rf /`` in a small directive prompt is a
      smoking gun, not prose.
    - When the assembled prompt includes runtime-loaded content — skill
      markdown (``has_skills=True``) or DATA injected from a job script's
      stdout / an upstream job's output (``has_injected_data=True``) — the
      LOOSER ``_scan_cron_skill_assembled`` pattern set is used: only
      unambiguous prompt-injection directives block; command-shape
      patterns are dropped and invisible unicode is sanitized (stripped +
      logged) rather than blocked, to avoid false-positives that
      permanently kill a job. Skill bodies are vetted at install time by
      ``skills_guard.py``; script output is produced by operator-authored
      code, the same trust class — and data feeds (e.g. a triage bot
      ingesting bug reports) legitimately quote dangerous commands.

    When the looser tier is selected because of injected data only,
    ``user_prompt`` (the raw, pre-assembly prompt) is additionally scanned
    with the STRICT set so the user-authored surface keeps the full
    create/update-time guarantee at runtime (defense-in-depth for legacy
    jobs that predate the create-time scanner).
    """
    from tools.cronjob_tools import _scan_cron_prompt, _scan_cron_skill_assembled

    if has_skills or has_injected_data:
        # Runtime-loaded content (vetted skill markdown and/or data from
        # operator-authored scripts) legitimately contains command-shape
        # strings. Invisible unicode is sanitized (not blocked) so a stray
        # zero-width space can't permanently kill the job; the cleaned
        # prompt is what actually runs.
        cleaned, scan_error = _scan_cron_skill_assembled(assembled)
        assembled = cleaned
        if not scan_error and not has_skills and user_prompt:
            # Data-injection path: keep the strict guarantee on the
            # user-authored prompt itself.
            scan_error = _scan_cron_prompt(user_prompt)
    else:
        scan_error = _scan_cron_prompt(assembled)
    if scan_error:
        job_label = job.get("name") or job.get("id") or "<unknown>"
        logger.warning(
            "Cron job '%s': assembled prompt blocked by injection scanner — %s",
            job_label,
            scan_error,
        )
        raise CronPromptInjectionBlocked(scan_error)
    return assembled


def _guard_job_credential_exfil(job: dict) -> None:
    """Fail closed if a job's stored provider/base_url pair would exfiltrate a
    credential (F8 runtime backstop; CWE-200/CWE-522).

    The model-callable cron tool validates this on create/update, but a job
    persisted before that guard — or written directly to the jobs store —
    reaches the scheduler's provider-resolution sink unchecked. Re-validate the
    EFFECTIVE stored pair with the same guard the tool uses, so a named
    provider's stored key is never paired with an off-host base_url at fire
    time. Raises ``RuntimeError`` (caught by the run_job failure path → the run
    is aborted and reported) when the pair is unsafe; returns ``None`` otherwise.

    Fallback providers come from operator config, not the model-callable job, so
    they are trusted and validated by the caller, not here.
    """
    try:
        from tools.cronjob_tools import _validate_cron_base_url
        err = _validate_cron_base_url(job.get("provider"), job.get("base_url"))
    except Exception as exc:
        # Fail CLOSED: this is the last guard before provider resolution, so an
        # unexpected validator/import error must not silently allow an unvetted
        # pair through. A job that carries no base_url override cannot exfiltrate
        # a stored credential via this path (there is nothing to validate, and
        # the validator would return None), so it still runs — that keeps the
        # overwhelmingly-common no-override jobs from wedging on an unrelated
        # error. But any job that DID set a base_url is refused until the
        # validator can actually vet the pair. Operator fallback providers come
        # from config, not the job, so they are unaffected.
        if job.get("base_url"):
            err = (
                f"could not validate provider/base_url pair "
                f"({exc.__class__.__name__}: {exc}); refusing to run a job with "
                "an unverified base_url override"
            )
        else:
            err = None
    if err:
        job_id = job.get("id")
        logger.error(
            "Job '%s': refusing to run — unsafe provider/base_url pair could "
            "exfiltrate a stored credential: %s",
            job_id, err,
        )
        raise RuntimeError(f"Cron job '{job_id}' blocked for safety: {err}")


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    job_id = job["id"]
    job_name = str(job.get("name") or job.get("prompt") or job_id or "cron job")

    # ---------------------------------------------------------------
    # no_agent short-circuit — the script IS the job, no LLM involvement.
    # ---------------------------------------------------------------
    # This mirrors the classic "run a bash script on a timer, send its
    # stdout to telegram" watchdog pattern. The agent path is skipped
    # entirely: no AIAgent, no prompt, no tool loop, no token spend.
    #
    # We check this BEFORE importing run_agent / constructing SessionDB so
    # a pure-script tick never pays for the agent machinery it isn't going
    # to use. Keep this block self-contained.
    #
    # Semantics:
    #   - script stdout (trimmed) → delivered verbatim as the final message
    #   - empty stdout            → silent run (no delivery, success=True)
    #   - non-zero exit / timeout → delivered as an error alert, success=False
    #   - wakeAgent=false gate    → treated like empty stdout (silent), since
    #                               the whole point of no_agent is that there
    #                               is no agent to wake
    if job.get("no_agent"):
        script_path = job.get("script")
        if not script_path:
            err = "no_agent=True but no script is set for this job"
            logger.error("Job '%s': %s", job_id, err)
            return False, "", "", err

        # Apply workdir if configured — lets scripts use predictable relative
        # paths. For no_agent jobs this is just the subprocess cwd (not an
        # agent TERMINAL_CWD bridge).
        _job_workdir = (job.get("workdir") or "").strip() or None
        _prior_cwd = None
        if _job_workdir and Path(_job_workdir).is_dir():
            _prior_cwd = os.getcwd()
            try:
                os.chdir(_job_workdir)
            except OSError:
                _prior_cwd = None

        try:
            ok, output = _run_job_script(script_path)
        finally:
            if _prior_cwd is not None:
                try:
                    os.chdir(_prior_cwd)
                except OSError:
                    pass

        now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")

        if not ok:
            # Script crashed / timed out / exited non-zero.  Deliver the
            # error so the user knows the watchdog itself broke — silent
            # failure for an alerting job is the worst-case outcome.
            alert = (
                f"⚠ Cron watchdog '{job_name}' script failed\n\n"
                f"{output}\n\n"
                f"Time: {now_iso}"
            )
            doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {now_iso}\n"
                f"**Mode:** no_agent (script)\n"
                f"**Status:** script failed\n\n"
                f"{output}\n"
            )
            return False, doc, alert, output

        # Honour the wakeAgent gate as a silent signal — `wakeAgent: false`
        # means "nothing to report this tick", same as empty stdout.
        if not _parse_wake_gate(output):
            logger.info(
                "Job '%s' (no_agent): wakeAgent=false gate — silent run", job_id
            )
            silent_doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {now_iso}\n"
                f"**Mode:** no_agent (script)\n"
                f"**Status:** silent (wakeAgent=false)\n"
            )
            return True, silent_doc, SILENT_MARKER, None

        if not output.strip():
            logger.info("Job '%s' (no_agent): empty stdout — silent run", job_id)
            silent_doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {now_iso}\n"
                f"**Mode:** no_agent (script)\n"
                f"**Status:** silent (empty output)\n"
            )
            return True, silent_doc, SILENT_MARKER, None

        doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {now_iso}\n"
            f"**Mode:** no_agent (script)\n\n"
            f"---\n\n"
            f"{output}\n"
        )
        return True, doc, output, None

    # ---------------------------------------------------------------
    # Default (LLM) path — import and construct the agent machinery now
    # that we know we actually need it. Doing these imports here instead of
    # at module top keeps no_agent ticks from paying for AIAgent / SessionDB
    # construction costs.
    # ---------------------------------------------------------------
    from run_agent import AIAgent

    # Initialize SQLite session store so cron job messages are persisted
    # and discoverable via session_search (same pattern as gateway/run.py).
    _session_db = None
    try:
        from hermes_state import SessionDB
        _session_db = SessionDB()
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)

    # Wake-gate: if this job has a pre-check script, run it BEFORE building
    # the prompt so a ``{"wakeAgent": false}`` response can short-circuit
    # the whole agent run. We pass the result into _build_job_prompt so
    # the script is only executed once.
    prerun_script = None
    script_path = job.get("script")
    if script_path:
        prerun_script = _run_job_script(script_path)
        _ran_ok, _script_output = prerun_script
        if _ran_ok and not _parse_wake_gate(_script_output):
            logger.info(
                "Job '%s' (ID: %s): wakeAgent=false, skipping agent run",
                job_name, job_id,
            )
            silent_doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Script gate returned `wakeAgent=false` — agent skipped.\n"
            )
            return True, silent_doc, SILENT_MARKER, None

    try:
        prompt = _build_job_prompt(job, prerun_script=prerun_script)
    except CronPromptInjectionBlocked as block_exc:
        # Assembled prompt (user prompt + loaded skill content) tripped the
        # injection scanner. Refuse to run the agent this tick and surface
        # a clear failure to the operator so they see WHY the scheduled job
        # didn't run and can audit the offending skill.
        logger.warning(
            "Job '%s' (ID: %s): blocked by prompt-injection scanner — %s",
            job_name, job_id, block_exc,
        )
        blocked_doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Status:** BLOCKED\n\n"
            "The assembled prompt (user prompt + loaded skill content) tripped "
            "the cron injection scanner and the agent was NOT run.\n\n"
            f"**Scanner result:** {block_exc}\n\n"
            "Audit the skill(s) attached to this job for prompt-injection "
            "payloads or invisible-unicode markers. If the skill is legitimate "
            "and the match is a false positive, rephrase the content to avoid "
            "the threat pattern (`tools/cronjob_tools.py::_CRON_THREAT_PATTERNS`)."
        )
        return False, blocked_doc, "", str(block_exc)
    if prompt is None:
        logger.info("Job '%s': script produced no output, skipping AI call.", job_name)
        return True, "", SILENT_MARKER, None
    origin = _resolve_origin(job)
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    agent = None

    # Mark this as a cron session so the approval system can apply cron_mode.
    # This env var is process-wide and persists for the lifetime of the
    # scheduler process — every job this process runs is a cron job.
    os.environ["HERMES_CRON_SESSION"] = "1"

    # Use ContextVars for per-job session/delivery state so parallel jobs
    # don't clobber each other's targets (os.environ is process-global).
    from gateway.session_context import set_session_vars, clear_session_vars, _VAR_MAP

    # Cron execution is an internal scheduler context, not a live inbound
    # gateway message. Do not seed HERMES_SESSION_* contextvars from the
    # stored ``origin`` (which is delivery routing metadata, not a sender
    # identity). Several tool consumers branch on these vars during job
    # execution and would otherwise behave as if a real user from the
    # origin chat was driving the agent:
    #   - tools/terminal_tool.py: background-process notification routing
    #     (notify_on_complete / watch_patterns) reads HERMES_SESSION_PLATFORM
    #     and HERMES_SESSION_CHAT_ID to populate watcher_platform / chat_id,
    #     which would route completion notifications to the origin chat
    #     instead of via HERMES_CRON_AUTO_DELIVER_* below.
    #   - tools/tts_tool.py: picks Opus vs MP3 based on
    #     HERMES_SESSION_PLATFORM == "telegram".
    #   - tools/skills_tool.py + agent/prompt_builder.py: per-platform
    #     skill-disable lists and the system-prompt cache key both consume
    #     HERMES_SESSION_PLATFORM.
    #   - tools/send_message_tool.py: mirror source labelling and the
    #     send_message gate read HERMES_SESSION_PLATFORM.
    # Cron output delivery itself reads job["origin"] directly via
    # _resolve_origin(job) and the HERMES_CRON_AUTO_DELIVER_* vars set
    # below, so clearing HERMES_SESSION_* here does not affect delivery.
    _ctx_tokens = set_session_vars(
        platform="",
        chat_id="",
        chat_name="",
    )
    _cron_delivery_vars = (
        "HERMES_CRON_AUTO_DELIVER_PLATFORM",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
        "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
    )
    for _var_name in _cron_delivery_vars:
        _VAR_MAP[_var_name].set("")

    # Per-job working directory.  When set (and validated at create/update
    # time), we point TERMINAL_CWD at it so:
    #   - build_context_files_prompt() picks up AGENTS.md / CLAUDE.md /
    #     .cursorrules from the job's project dir, AND
    #   - the terminal, file, and code-exec tools run commands from there.
    #
    # os.environ["TERMINAL_CWD"] is process-global, so this override is
    # serialized by _terminal_cwd_lock (acquired just below): a workdir job
    # holds it as a writer for its whole run, excluding every other job, while
    # workdir-less jobs hold it as readers and stay parallel with each other.
    # The sequential pool only keeps workdir jobs from overlapping EACH OTHER;
    # the lock is what additionally keeps a concurrently-firing workdir-less
    # parallel-pool job from observing this override and running its shell /
    # file / code-exec commands in the wrong directory.  For workdir-less jobs
    # we leave TERMINAL_CWD untouched — preserves the original behaviour
    # (skip_context_files=True, tools use whatever cwd the scheduler has).
    _job_workdir = (job.get("workdir") or "").strip() or None
    if _job_workdir and not Path(_job_workdir).is_dir():
        # Directory was removed between create-time validation and now.  Log
        # and drop back to old behaviour rather than crashing the job.
        logger.warning(
            "Job '%s': configured workdir %r no longer exists — running without it",
            job_id, _job_workdir,
        )
        _job_workdir = None

    # Snapshot the current env value BEFORE acquiring the lock so the finally
    # below can always restore it, even if an exception fires before we set the
    # override inside the try.  This read can't leak the lock (it precedes the
    # acquire) and is a no-op for workdir-less jobs (they never mutate the env).
    _prior_terminal_cwd = os.environ.get("TERMINAL_CWD", "_UNSET_")

    _holds_cwd_write = _job_workdir is not None
    if _holds_cwd_write:
        _terminal_cwd_lock.acquire_write()
    else:
        _terminal_cwd_lock.acquire_read()

    # Everything after the acquire MUST live inside this try, so the finally
    # below always releases the lock even if the env override or any later
    # statement raises.  A leaked writer would deadlock the whole scheduler
    # (every future job blocks on acquire_*); a leaked reader blocks all
    # future writers.  Acquire itself can't leak (it either blocks or returns).
    try:
        if _job_workdir:
            os.environ["TERMINAL_CWD"] = _job_workdir
            logger.info("Job '%s': using workdir %s", job_id, _job_workdir)

        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart. Route through
        # load_hermes_dotenv (not a bare load_dotenv) and reset the secret-
        # source cache first: startup already applied external secrets and
        # recorded this HERMES_HOME in _APPLIED_HOMES, so a naive reload would
        # re-apply only the .env placeholder and never re-resolve a Bitwarden/
        # BSM-backed secret — leaving cron jobs 401'ing on the placeholder
        # (#33465). Clearing the cache forces the re-pull; the resolved secret
        # overrides the placeholder only when secrets.bitwarden.override_existing
        # is set (mirrors startup), and the Bitwarden value-cache keeps the
        # forced re-pull off the network. load_hermes_dotenv also handles the
        # utf-8/latin-1 encoding fallback internally.
        from hermes_cli.env_loader import (
            load_hermes_dotenv,
            reset_secret_source_cache,
        )
        reset_secret_source_cache()
        load_hermes_dotenv(hermes_home=_get_hermes_home())

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set(delivery_target["platform"])
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set(str(delivery_target["chat_id"]))
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set(
                ""
                if delivery_target.get("thread_id") is None
                else str(delivery_target["thread_id"])
            )

        # Model resolution precedence: per-job override > HERMES_MODEL env >
        # config.yaml ``model:`` (string or ``{default: ...}``). The per-job
        # value is intentionally re-read from storage every tick so a
        # ``cronjob action=update model=...`` after a failed run takes effect
        # on the next tick — there is no in-memory cache.
        model = job.get("model") or os.getenv("HERMES_MODEL") or ""

        # Load config.yaml for model, reasoning, prefill, toolsets, provider routing
        _cfg = {}
        try:
            import yaml
            _cfg_path = str(_get_hermes_home() / "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path, encoding="utf-8") as _f:
                    _cfg = yaml.safe_load(_f) or {}
                # Managed scope: a scheduled job must honor administrator-pinned
                # model / reasoning / toolsets / provider_routing too. This loader
                # builds its own dict, so overlay managed values via the shared
                # helper (fail-open, no-op when no managed scope).
                try:
                    from hermes_cli import managed_scope
                    _cfg = managed_scope.apply_managed_overlay(_cfg)
                except Exception:
                    pass
                _cfg = _expand_env_vars(_cfg)
                # Coerce null/missing to {} so a falsy default never
                # clobbers an already-resolved env value with ``None``.
                _model_cfg = _cfg.get("model") or {}
                if not job.get("model"):
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        # Mirror the CLI/oneshot resolution: prefer ``default``,
                        # accept a ``model`` alias, overwrite only when truthy.
                        _default = _model_cfg.get("default") or _model_cfg.get("model")
                        if _default:
                            model = _default
        except Exception as e:
            logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

        # Fail fast if no model resolved from job / env / config.yaml: an empty
        # model otherwise reaches the provider as an opaque 400 (#23979).
        if not (isinstance(model, str) and model.strip()):
            raise RuntimeError(
                f"Cron job '{job_name}' has no model configured "
                f"(job.model={job.get('model')!r}, "
                f"HERMES_MODEL={os.getenv('HERMES_MODEL', '')!r}, "
                "config.yaml model.default missing or empty). "
                f"Set a per-job model via "
                f"`cronjob action=update job_id={job_id} model=<name>` or set a "
                "default with `hermes model <name>`."
            )

        # Apply IPv4 preference if configured.
        try:
            from hermes_constants import apply_ipv4_preference
            _net_cfg = _cfg.get("network", {})
            if isinstance(_net_cfg, dict) and _net_cfg.get("force_ipv4"):
                apply_ipv4_preference(force=True)
        except Exception:
            pass

        # Reasoning config from config.yaml
        from hermes_constants import parse_reasoning_effort
        effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)

        # Prefill messages from env or config.yaml. The top-level
        # prefill_messages_file key is canonical; agent.prefill_messages_file is
        # retained as a legacy fallback for older CLI/godmode configs.
        prefill_messages = None
        agent_cfg = _cfg.get("agent", {}) if isinstance(_cfg.get("agent", {}), dict) else {}
        prefill_file = (
            os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
            or _cfg.get("prefill_messages_file", "")
            or agent_cfg.get("prefill_messages_file", "")
        )
        if prefill_file:
            pfpath = Path(prefill_file).expanduser()
            if not pfpath.is_absolute():
                pfpath = _get_hermes_home() / pfpath
            if pfpath.exists():
                try:
                    with open(pfpath, "r", encoding="utf-8") as _pf:
                        prefill_messages = json.load(_pf)
                    if not isinstance(prefill_messages, list):
                        prefill_messages = None
                except Exception as e:
                    logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
                    prefill_messages = None

        # Max iterations
        max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90

        # Provider routing
        pr = _cfg.get("provider_routing") or {}

        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )
        from hermes_cli.auth import AuthError

        # F8 runtime backstop: never resolve a stored provider/base_url pair that
        # would ship a named provider's stored credential to an off-host endpoint
        # (CWE-200/CWE-522). The cron tool validates this on create/update, but a
        # job persisted before that guard — or written directly to the jobs store
        # — reaches this sink unchecked. Fail closed before resolution so no
        # off-host call is ever made with a stored key.
        _guard_job_credential_exfil(job)

        try:
            # Do not inject HERMES_INFERENCE_PROVIDER here. resolve_runtime_provider()
            # already prefers persisted config over stale shell/env overrides when
            # no explicit provider is requested. Passing the env var here short-
            # circuits that precedence and can resurrect old providers (for
            # example DeepSeek) for cron jobs that do not pin provider/model.
            runtime_kwargs = {
                "requested": job.get("provider"),
            }
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except AuthError as auth_exc:
            # Primary provider auth failed — try fallback chain before giving up.
            logger.warning("Job '%s': primary auth failed (%s), trying fallback", job_id, auth_exc)
            fb_list = get_fallback_chain(_cfg)
            runtime = None
            for entry in fb_list:
                try:
                    fb_kwargs = {"requested": entry.get("provider")}
                    if entry.get("base_url"):
                        fb_kwargs["explicit_base_url"] = entry["base_url"]
                    if entry.get("api_key"):
                        fb_kwargs["explicit_api_key"] = entry["api_key"]
                    runtime = resolve_runtime_provider(**fb_kwargs)
                    logger.info("Job '%s': fallback resolved to %s", job_id, runtime.get("provider"))
                    break
                except Exception as fb_exc:
                    logger.debug("Job '%s': fallback %s failed: %s", job_id, entry.get("provider"), fb_exc)
            if runtime is None:
                raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            raise RuntimeError(message) from exc

        # Provider/model-drift fail-closed guard (#44585).
        #
        # An UNPINNED job (no explicit job["provider"]/["model"]) follows the
        # global default, which can change after the job was created — a switch
        # to a paid PROVIDER (e.g. nous) OR a paid MODEL on the same provider
        # (e.g. claude-fable-5 on openrouter). Without a guard the job would
        # silently inherit that change and spend real money on every tick — the
        # $7.73 incident named BOTH a provider and a model.
        #
        # create_job() snapshots whatever resolution would have picked at
        # creation for each unpinned axis (job["provider_snapshot"] /
        # job["model_snapshot"]). Here, for each axis that (a) has a snapshot and
        # (b) is unpinned and (c) currently resolves to a DIFFERENT value, we
        # fail closed: skip this run, make NO paid call, and deliver a loud,
        # actionable alert telling the user to pin the axis explicitly.
        #
        # Back-compat: an axis with no snapshot (pre-existing jobs, no_agent, or
        # any axis whose creation-time resolution failed) behaves exactly as
        # before — the guard never engages for it. Pinned axes are unaffected.
        _drift: list[str] = []
        _provider_snapshot = (job.get("provider_snapshot") or "").strip().lower()
        if _provider_snapshot and not (job.get("provider") or "").strip():
            _current_provider = str(runtime.get("provider") or "").strip().lower()
            if _current_provider and _current_provider != _provider_snapshot:
                _drift.append(
                    f"provider '{_provider_snapshot}' -> '{_current_provider}'"
                )
        _model_snapshot = (job.get("model_snapshot") or "").strip().lower()
        if _model_snapshot and not (job.get("model") or "").strip():
            _current_model = str(model or "").strip().lower()
            if _current_model and _current_model != _model_snapshot:
                _drift.append(
                    f"model '{_model_snapshot}' -> '{_current_model}'"
                )
        if _drift:
            _changes = "; ".join(_drift)
            logger.warning(
                "Job '%s': SKIPPED — global inference config drifted since "
                "creation (%s) and this job is unpinned. Skipped to prevent "
                "unintended spend. Pin explicitly to proceed: "
                "`cronjob action=update job_id=%s provider=<p> model=<m>`.",
                job_id,
                _changes,
                job_id,
            )
            raise RuntimeError(
                f"Skipped to prevent unintended spend: global inference config "
                f"drifted since this job was created ({_changes}), and this job "
                f"is unpinned. No inference call was made. To run on the new "
                f"config, pin it explicitly: `cronjob action=update "
                f"job_id={job_id} provider=<provider> model=<model>` "
                f"(or pin the original values to keep them). See #44585."
            )

        fallback_model = get_fallback_chain(_cfg) or None
        credential_pool = None
        runtime_provider = str(runtime.get("provider") or "").strip().lower()
        if runtime_provider:
            try:
                from agent.credential_pool import load_pool
                pool = load_pool(runtime_provider)
                if pool.has_credentials():
                    credential_pool = pool
                    logger.info(
                        "Job '%s': loaded credential pool for provider %s with %d entries",
                        job_id,
                        runtime_provider,
                        len(pool.entries()),
                    )
            except Exception as e:
                logger.debug("Job '%s': failed to load credential pool for %s: %s", job_id, runtime_provider, e)

        # Initialize MCP servers so configured mcp_servers are available to
        # the agent's tool registry before AIAgent is constructed. Without
        # this, cron jobs never saw any MCP tools — only the gateway / CLI
        # paths called discover_mcp_tools() at startup. Idempotent: subsequent
        # ticks short-circuit on already-connected servers inside
        # register_mcp_servers(). Non-fatal on failure: a broken MCP server
        # shouldn't kill an otherwise-working cron job. See #4219.
        try:
            from tools.mcp_tool import discover_mcp_tools
            _mcp_tools = discover_mcp_tools()
            if _mcp_tools:
                logger.info(
                    "Job '%s': %d MCP tool(s) available",
                    job_id, len(_mcp_tools),
                )
        except Exception as _mcp_exc:
            logger.warning(
                "Job '%s': MCP initialization failed (non-fatal): %s",
                job_id, _mcp_exc,
            )

        agent = AIAgent(
            model=model,
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            api_mode=runtime.get("api_mode"),
            acp_command=runtime.get("command"),
            acp_args=runtime.get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            fallback_model=fallback_model,
            credential_pool=credential_pool,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            openrouter_min_coding_score=(_cfg.get("openrouter") or {}).get("min_coding_score"),
            enabled_toolsets=_resolve_cron_enabled_toolsets(job, _cfg),
            disabled_toolsets=_resolve_cron_disabled_toolsets(_cfg),
            quiet_mode=True,
            # Cron jobs should always inherit the user's SOUL.md identity from
            # HERMES_HOME. When a workdir is configured, also inject project
            # context files (AGENTS.md / CLAUDE.md / .cursorrules) from there.
            # Without a workdir, keep cwd context discovery disabled.
            skip_context_files=not bool(_job_workdir),
            load_soul_identity=True,
            skip_memory=True,  # Cron system prompts would corrupt user representations
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
        )
        
        # Run the agent with an *inactivity*-based timeout: the job can run
        # for hours if it's actively calling tools / receiving stream tokens,
        # but a hung API call or stuck tool with no activity for the configured
        # duration is caught and killed.  Default 600s (10 min inactivity);
        # override via HERMES_CRON_TIMEOUT env var.  0 = unlimited.
        #
        # Uses the agent's built-in activity tracker (updated by
        # _touch_activity() on every tool call, API call, and stream delta).
        _raw_cron_timeout = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
        if _raw_cron_timeout:
            try:
                _cron_timeout = float(_raw_cron_timeout)
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid HERMES_CRON_TIMEOUT=%r; using default 600s",
                    _raw_cron_timeout,
                )
                _cron_timeout = 600.0
        else:
            _cron_timeout = 600.0
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        _POLL_INTERVAL = 5.0
        _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Preserve scheduler-scoped ContextVar state (for example skill-declared
        # env passthrough registrations) when the cron run hops into the worker
        # thread used for inactivity timeout monitoring.
        _cron_context = contextvars.copy_context()
        _cron_future = _cron_pool.submit(_cron_context.run, agent.run_conversation, prompt)
        _inactivity_timeout = False
        try:
            if _cron_inactivity_limit is None:
                # Unlimited — just wait for the result.
                result = _cron_future.result()
            else:
                result = None
                while True:
                    done, _ = concurrent.futures.wait(
                        {_cron_future}, timeout=_POLL_INTERVAL,
                    )
                    if done:
                        result = _cron_future.result()
                        break
                    # Agent still running — check inactivity.
                    _idle_secs = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            _act = agent.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    if _idle_secs >= _cron_inactivity_limit:
                        _inactivity_timeout = True
                        break
        except Exception:
            _cron_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            _cron_pool.shutdown(wait=False, cancel_futures=True)

        if _inactivity_timeout:
            # Build diagnostic summary from the agent's activity tracker.
            _activity = {}
            if hasattr(agent, "get_activity_summary"):
                try:
                    _activity = agent.get_activity_summary()
                except Exception:
                    pass
            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                job_name, _secs_ago, _cron_inactivity_limit,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )
            if hasattr(agent, "interrupt"):
                agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(
                f"Cron job '{job_name}' idle for "
                f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
                f"— last activity: {_last_desc}"
            )

        # Guard against non-dict returns from run_conversation under error conditions
        if not isinstance(result, dict):
            raise RuntimeError(
                f"agent.run_conversation returned {type(result).__name__} instead of dict: {result!r}"
            )

        # If the agent itself reported failure (e.g. all retries exhausted on
        # API errors, model abort, mid-run interrupt), do not silently mark the
        # job as successful. run_agent populates `failed=True`/`completed=False`
        # on these paths and may put the error into `final_response`, which
        # would otherwise be delivered as if it were the agent's reply and the
        # job's `last_status` set to "ok". Raise so the except handler below
        # builds the proper failure tuple. (issue #17855)
        turn_exit_reason = str(result.get("turn_exit_reason") or "")
        final_response_text = (result.get("final_response") or "").strip()
        max_iteration_summary = (
            result.get("failed") is not True
            and result.get("completed") is False
            and turn_exit_reason.startswith("max_iterations_reached(")
            and bool(final_response_text)
        )
        if result.get("failed") is True or (result.get("completed") is False and not max_iteration_summary):
            _err_text = (
                result.get("error")
                or final_response_text
                or "agent reported failure"
            )
            raise RuntimeError(_err_text)
        if max_iteration_summary:
            logger.warning(
                "Job '%s' reached the iteration limit but produced a final fallback response; "
                "delivering the response instead of failing the cron run",
                job_name,
            )

        final_response = result.get("final_response", "") or ""
        # Strip leaked placeholder text that upstream may inject on empty completions.
        if final_response.strip() == "(No response generated)":
            final_response = ""
        # Cron silence on abnormal empty turns.  The turn-completion explainer
        # (#34452) replaces a blank/empty model turn with a "⚠️ No reply: …"
        # string so interactive surfaces (CLI/gateway) explain why the box is
        # empty.  In a cron context that turns a previously-silent empty turn
        # into a delivered warning (Manfredi's Telegram symptom).  Detect the
        # explainer text deterministically (via the same formatter that
        # produced it) and treat it as empty so the empty-response suppression
        # and soft-failure marking below apply — restoring pre-#34452 silence
        # for scheduled jobs without disabling the explainer everywhere.
        if final_response.strip() and turn_exit_reason:
            try:
                _explainer_text = AIAgent._format_turn_completion_explanation(turn_exit_reason)
            except Exception:
                _explainer_text = ""
            if _explainer_text and final_response.strip() == _explainer_text.strip():
                logger.info(
                    "Job '%s': abnormal empty turn (%s) — suppressing explainer for cron delivery",
                    job_id,
                    turn_exit_reason,
                )
                final_response = ""
        # Use a separate variable for log display; keep final_response clean
        # for delivery logic (empty response = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}
{_format_model_provider_line(model, runtime.get('provider'))}

## Prompt

{prompt}

## Response

{logged_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}
{_format_model_provider_line(model, runtime.get('provider'))}

## Prompt

{prompt}

## Error

```
{error_msg}
```
"""
        return False, output, "", error_msg

    finally:
        # Restore TERMINAL_CWD to whatever it was before this job ran.  We
        # only ever mutate it when the job has a workdir; see the setup block
        # at the top of run_job for the serialization guarantee.
        if _job_workdir:
            if _prior_terminal_cwd == "_UNSET_":
                os.environ.pop("TERMINAL_CWD", None)
            else:
                os.environ["TERMINAL_CWD"] = _prior_terminal_cwd
        # Release the cwd lock now that the env is restored, so a waiting
        # workdir job (or queued reader) can proceed without seeing the override.
        if _holds_cwd_write:
            _terminal_cwd_lock.release_write()
        else:
            _terminal_cwd_lock.release_read()
        # Clean up ContextVar session/delivery state for this job.
        clear_session_vars(_ctx_tokens)
        for _var_name in _cron_delivery_vars:
            _VAR_MAP[_var_name].set("")
        if _session_db:
            # Title the cron session from the job (name → short prompt → id) so
            # sidebars/history show a meaningful label instead of the injected
            # "[IMPORTANT: …]" hint that is the session's first message. Set here
            # (not at create time) so the agent's own INSERT keeps model /
            # system_prompt; this only UPDATEs the title column. The run-time
            # suffix keeps it unique against the sessions.title index across runs.
            try:
                _title_base = " ".join(job_name.split())[:60].strip() or f"cron {job_id}"
                _cron_title = f"{_title_base} · {_hermes_now().strftime('%b %d %H:%M')}"
                _session_db.set_session_title(_cron_session_id, _cron_title)
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to set cron session title: %s", job_id, e)
            try:
                _session_db.end_session(_cron_session_id, "cron_complete")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to end session: %s", job_id, e)
            try:
                _session_db.close()
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)
        # Release subprocesses, terminal sandboxes, browser daemons, and the
        # main OpenAI/httpx client held by this ephemeral cron agent. Without
        # this, a gateway that ticks cron every N minutes leaks fds per job
        # until it hits EMFILE (#10200 / "too many open files").
        try:
            if agent is not None:
                agent.close()
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Job '%s': failed to close agent resources: %s", job_id, e)
        # Each cron run spins up a short-lived worker thread whose event loop
        # dies as soon as the ``ThreadPoolExecutor`` shuts down. Any async
        # httpx clients cached under that loop are now unusable — reap them
        # so their transports don't accumulate in the process-global cache.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception as e:
            logger.debug("Job '%s': failed to reap stale auxiliary clients: %s", job_id, e)


def run_one_job(job: dict, *, adapters=None, loop=None, verbose: bool = False) -> bool:
    """Run ONE due job end-to-end: execute → save output → deliver → mark.

    This is the shared firing body extracted from ``tick``'s per-job closure so
    that BOTH the built-in ticker and an external provider's ``fire_due`` (e.g.
    Chronos) run the identical sequence — no duplicated correctness.

    It does NOT decide whether the job is due, claim it, or compute the next
    run — those are the caller's concern (``tick`` advances ``next_run_at``
    under the file lock before dispatch; an external provider claims via the
    store CAS). This function only fires the given job once.

    Returns True if the job was processed (even if the job itself failed —
    failure is recorded via ``mark_job_run``), False only if processing raised.
    """
    try:
        # Pre-run dispatch claim (issue #38758): atomically commit a finite
        # one-shot's dispatch BEFORE its side effect runs, so a tick that dies
        # mid-execution (gateway kill, OOM, segfault, hard-timeout) cannot
        # re-fire the job forever on restart. No-op for recurring jobs (they
        # use advance_next_run) and infinite/no-repeat jobs. This lives here in
        # the shared body so BOTH the built-in ticker and the external provider
        # (Chronos fire_due) get at-most-times semantics.
        if not claim_dispatch(job["id"]):
            logger.info(
                "Job '%s': one-shot dispatch limit reached — skipping",
                job.get("name", job["id"]),
            )
            return True  # not an error — already handled/removed

        success, output, final_response, error = run_job(job)

        output_file = save_job_output(job["id"], output)
        if verbose:
            logger.info("Output saved to: %s", output_file)

        # Deliver the final response to the origin/target chat.
        # If the agent responded with [SILENT], skip delivery (but
        # output is already saved above).  Failed jobs always deliver.
        deliver_content = final_response if success else _summarize_cron_failure_for_delivery(job, error)
        # Treat whitespace-only final responses the same as empty
        # responses: do not deliver a blank message, and let the
        # empty-response guard below mark the run as a soft failure.
        should_deliver = bool(deliver_content.strip())
        # Cron silence suppression — see _is_cron_silence_response.  Replaces the
        # old `SILENT_MARKER in ...upper()` substring check, which both leaked
        # bracketless near-markers ("SILENT" / "NO_REPLY") and wrongly swallowed
        # a real report that merely quoted "[SILENT]" mid-sentence (#51438,
        # #46917).  Keeps the intentional bracketed-prefix / trailing-line
        # tolerance the cron contract relies on.
        if should_deliver and success and _is_cron_silence_response(deliver_content):
            logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
            should_deliver = False

        delivery_error = None
        if should_deliver:
            try:
                delivery_error = _deliver_result(job, deliver_content, adapters=adapters, loop=loop)
            except Exception as de:
                delivery_error = str(de)
                logger.error("Delivery failed for job %s: %s", job["id"], de)

        # Treat empty final_response as a soft failure so last_status
        # is not "ok" — the agent ran but produced nothing useful.
        # (issue #8585)
        if success and not final_response.strip():
            success = False
            error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"

        mark_job_run(job["id"], success, error, delivery_error=delivery_error)
        return True

    except Exception as e:
        logger.error("Error processing job %s: %s", job['id'], e)
        mark_job_run(job["id"], False, str(e))
        return False


def _notify_provider_jobs_changed() -> None:
    """Best-effort: tell the active scheduler provider the job set changed.

    Called by the consumer surfaces (model tool / CLI / REST) AFTER a
    successful store mutation (create/update/remove/pause/resume) so an external
    provider (Chronos) can re-provision/cancel the affected one-shot via NAS.
    No-op for the built-in (it re-reads jobs.json each tick), so the default
    path is unchanged. Lives here (not in cron/jobs.py) to keep the store free
    of provider imports — avoids an import cycle and keeps jobs.py low-coupling.
    Never raises into the caller.
    """
    try:
        from cron.scheduler_provider import resolve_cron_scheduler
        resolve_cron_scheduler().on_jobs_changed()
    except Exception as e:
        logger.debug("on_jobs_changed notify failed: %s", e)


def tick(verbose: bool = True, adapters=None, loop=None, sync: bool = True) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
        adapters: Optional dict mapping Platform → live adapter (from gateway)
        loop: Optional asyncio event loop (from gateway) for live adapter sends
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    lock_dir, lock_file = _get_lock_paths()
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    lock_fd = None
    try:
        lock_fd = open(lock_file, "w", encoding="utf-8")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        if lock_fd is not None:
            lock_fd.close()
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        # Advance next_run_at for all recurring jobs FIRST, under the file lock,
        # before any execution begins.  This preserves at-most-once semantics.
        # For parallel jobs that are already running, advance_next_run keeps
        # bumping next_run_at forward so the grace window never expires.
        # mark_job_run() overwrites next_run_at on completion.
        for job in due_jobs:
            advance_next_run(job["id"])

        # Resolve max parallel workers: env var > config.yaml > unbounded.
        # Set HERMES_CRON_MAX_PARALLEL=1 to restore old serial behaviour.
        _max_workers: Optional[int] = None
        try:
            _env_par = os.getenv("HERMES_CRON_MAX_PARALLEL", "").strip()
            if _env_par:
                _max_workers = int(_env_par) or None
        except (ValueError, TypeError):
            logger.warning("Invalid HERMES_CRON_MAX_PARALLEL value; defaulting to unbounded")
        if _max_workers is None:
            try:
                _ucfg = load_config() or {}
                _cfg_par = (
                    _ucfg.get("cron", {}) if isinstance(_ucfg, dict) else {}
                ).get("max_parallel_jobs")
                if _cfg_par is not None:
                    _max_workers = int(_cfg_par) or None
            except Exception:
                pass

        if verbose:
            logger.info(
                "Running %d job(s) in parallel (max_workers=%s)",
                len(due_jobs),
                _max_workers if _max_workers else "unbounded",
            )

        def _process_job(job: dict) -> bool:
            """Run one due job end-to-end. Thin wrapper around the shared
            module-level ``run_one_job`` so ``tick`` and external providers
            (Chronos ``fire_due``) use the identical execute→save→deliver→mark
            body."""
            return run_one_job(job, adapters=adapters, loop=loop, verbose=verbose)

        # Partition due jobs: those with a per-job workdir mutate
        # os.environ["TERMINAL_CWD"] inside run_job, which is process-global, so
        # they queue on the single-thread sequential pool to run one at a time.
        # That alone only keeps workdir jobs from overlapping EACH OTHER;
        # run_job's _terminal_cwd_lock is what additionally stops a concurrently
        # firing workdir-less parallel-pool job from observing the override.
        sequential_jobs = [j for j in due_jobs if (j.get("workdir") or "").strip()]
        parallel_jobs = [j for j in due_jobs if not (j.get("workdir") or "").strip()]

        _results: list = []
        _all_futures: list = []

        def _submit_with_guard(job: dict, pool: concurrent.futures.ThreadPoolExecutor):
            """Submit a job fire-and-forget with the in-flight dedup guard.

            Returns the future, or None if the job was skipped because a prior
            tick's run of the same job is still in flight.  The running-set
            membership is released in the worker's finally block.
            """
            job_id = job["id"]
            with _running_lock:
                if job_id in _running_job_ids:
                    logger.info("Job '%s' already running — skipping", job.get("name", job_id))
                    return None
                _running_job_ids.add(job_id)
            _ctx = contextvars.copy_context()

            def _run_and_release(j=job, ctx=_ctx):
                try:
                    return ctx.run(_process_job, j)
                finally:
                    with _running_lock:
                        _running_job_ids.discard(j["id"])

            return pool.submit(_run_and_release)

        # Sequential pass for env-mutating (workdir) jobs.
        # Queued to a persistent single-thread pool so they run one at a time
        # WITHOUT blocking the ticker thread — a long workdir job no
        # longer starves the rest of the schedule (same fix as the parallel
        # pass, just serialized).  The in-flight guard prevents a still-running
        # job from being re-queued on the next tick.
        if sequential_jobs:
            seq_pool = _get_sequential_pool()
            for job in sequential_jobs:
                fut = _submit_with_guard(job, seq_pool)
                if fut is None:
                    continue
                _all_futures.append(fut)
                if not sync:
                    _results.append(True)  # optimistically counted

        # Parallel pass — persistent pool, non-blocking dispatch.
        # Jobs that are already running (from a previous tick) are skipped.
        # mark_job_run() updates next_run_at on completion, so the next tick
        # after completion finds the job due again naturally.  No catch-up
        # queue needed.
        if parallel_jobs:
            pool = _get_parallel_pool(_max_workers)
            for job in parallel_jobs:
                fut = _submit_with_guard(job, pool)
                if fut is None:
                    continue
                _all_futures.append(fut)
                if not sync:
                    _results.append(True)  # optimistically counted

        # Best-effort sweep of MCP stdio subprocesses that survived their
        # session teardown.  Must run AFTER jobs finish so active sessions
        # (including live user chats) are never touched — only PIDs explicitly
        # detected as orphans in tools.mcp_tool._run_stdio's finally block are
        # reaped.
        def _sweep_mcp_orphans() -> None:
            try:
                from tools.mcp_tool import _kill_orphaned_mcp_children
                _kill_orphaned_mcp_children()
            except Exception as _e:
                logger.debug("Post-tick MCP orphan cleanup failed: %s", _e)

        if sync:
            # Sync mode (tests / manual ticks): wait for all dispatched jobs,
            # collect results, then sweep once.
            for f in concurrent.futures.as_completed(_all_futures):
                try:
                    _results.append(f.result())
                except Exception as exc:
                    logger.error("Cron job future failed: %s", exc)
                    _results.append(False)
            _sweep_mcp_orphans()
            return sum(_results)

        # Async (gateway ticker) mode: don't block.  Sweep orphans via a
        # done-callback fired after the LAST dispatched job completes, so the
        # sweep still happens after jobs finish without stalling the tick.
        if _all_futures:
            _remaining = [len(_all_futures)]

            def _on_done(_f: concurrent.futures.Future) -> None:
                _remaining[0] -= 1
                try:
                    _exc = _f.exception()
                    if _exc is not None:
                        logger.error("Cron job future failed in async mode: %s", _exc, exc_info=(type(_exc), _exc, _exc.__traceback__))
                except Exception:
                    pass
                if _remaining[0] <= 0:
                    _sweep_mcp_orphans()

            for _f in _all_futures:
                _f.add_done_callback(_on_done)
        else:
            # Nothing dispatched (all skipped / no due jobs) — sweep inline.
            _sweep_mcp_orphans()

        return sum(_results)
    finally:
        if fcntl:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
