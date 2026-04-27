"""
Shell-script hooks bridge.

Reads the ``hooks:`` block from ``cli-config.yaml``, prompts the user for
consent on first use of each ``(event, command)`` pair, and registers
callbacks on the existing plugin hook manager so every existing
``invoke_hook()`` site dispatches to the configured shell scripts — with
zero changes to call sites.

Design notes
------------
* Python plugins and shell hooks compose naturally: both flow through
  :func:`hermes_cli.plugins.invoke_hook` and its aggregators.  Python
  plugins are registered first (via ``discover_and_load()``) so their
  block decisions win ties over shell-hook blocks.
* Subprocess execution uses ``shlex.split(os.path.expanduser(command))``
  with ``shell=False`` — no shell injection footguns.  Users that need
  pipes/redirection wrap their logic in a script.
* First-use consent is gated by the allowlist under
  ``~/.hermes/shell-hooks-allowlist.json``.  Non-TTY callers must pass
  ``accept_hooks=True`` (resolved from ``--accept-hooks``,
  ``HERMES_ACCEPT_HOOKS``, or ``hooks_auto_accept: true`` in config)
  for registration to succeed without a prompt.
* Registration is idempotent — safe to invoke from both the CLI entry
  point (``hermes_cli/main.py``) and the gateway entry point
  (``gateway/run.py``).

Wire protocol
-------------
**stdin** (JSON, piped to the script)::

    {
        "hook_event_name": "pre_tool_call",
        "tool_name":       "terminal",
        "tool_input":      {"command": "rm -rf /"},
        "session_id":      "sess_abc123",
        "cwd":             "/home/user/project",
        "extra":           {...}   # event-specific kwargs
    }

**stdout** (JSON, optional — anything else is ignored)::

    # Block a pre_tool_call (either shape accepted; normalised internally):
    {"decision": "block", "reason":  "Forbidden command"}   # Claude-Code-style
    {"action":   "block", "message": "Forbidden command"}   # Hermes-canonical

    # Inject context for pre_llm_call:
    {"context": "Today is Friday"}

    # Silent no-op:
    <empty or any non-matching JSON object>
"""

from __future__ import annotations

import atexit
import concurrent.futures
import difflib
import json
import logging
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

try:
    import fcntl  # POSIX only; Windows falls back to best-effort without flock.
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
ALLOWLIST_FILENAME = "shell-hooks-allowlist.json"

# Platform guard for POSIX-only process-group APIs used in the
# subprocess cleanup paths below.  Matches the convention in
# ``tools/process_registry.py`` and is enforced by
# ``tests/tools/test_windows_compat.py``.
_IS_WINDOWS = platform.system() == "Windows"

# Async-hook config clamps.  Both values are pulled live from
# ``cli-config.yaml`` so tests and operator overrides flow through
# without code change.
_DEFAULT_ASYNC_POOL_SIZE = 10
_DEFAULT_ASYNC_SHUTDOWN_GRACE = 5
_ASYNC_POOL_SIZE_BOUNDS = (1, 100)
_ASYNC_SHUTDOWN_GRACE_BOUNDS = (0, 60)

# Events where ``async: true`` registers with a warning because the
# return value is typically load-bearing but there are observational
# use cases (fire-and-forget logging per tool call).  Events not in
# this set and not in the reject set are silently accepted.
_WARN_ASYNC_EVENTS = frozenset({"pre_tool_call"})

# Events where ``async: true`` is a hard config error — the return
# value is the *only* product of the hook, so async would make the
# registration 100% useless.  Rejected at config-parse time.
_REJECT_ASYNC_EVENTS = frozenset({
    "pre_llm_call",
    "transform_tool_result",
    "transform_terminal_output",
})

# (event, matcher, command) triples that have been wired to the plugin
# manager in the current process.  Matcher is part of the key because
# the same script can legitimately register for different matchers under
# the same event (e.g. one entry per tool the user wants to gate).
# Second registration attempts for the exact same triple become no-ops
# so the CLI and gateway can both call register_from_config() safely.
_registered: Set[Tuple[str, Optional[str], str]] = set()
_registered_lock = threading.Lock()

# Intra-process lock for allowlist read-modify-write on platforms that
# lack ``fcntl`` (non-POSIX).  Kept separate from ``_registered_lock``
# because ``register_from_config`` already holds ``_registered_lock`` when
# it triggers ``_record_approval`` — reusing it here would self-deadlock
# (``threading.Lock`` is non-reentrant).  POSIX callers use the sibling
# ``.lock`` file via ``fcntl.flock`` and bypass this.
_allowlist_write_lock = threading.Lock()

# Live subprocess and future tracking for the shutdown / SIGINT path.
# Populated by ``_spawn`` (procs) and the async callback factory
# (futures); drained by ``shutdown_async_hooks`` and SIGINT handlers.
# CPython's GIL makes individual ``add`` / ``discard`` calls atomic, but
# iteration during shutdown must be locked against concurrent
# done_callback removal — hence the explicit lock.
_async_tracking_lock = threading.Lock()
_live_procs: Set[subprocess.Popen] = set()
_live_futures: Set[concurrent.futures.Future] = set()

# Async pool / semaphore singletons — created lazily on first use so
# tests and CLI/gateway startup don't eagerly spin up worker threads.
# Double-checked locking via ``_async_init_lock`` prevents two
# concurrent firings from racing through the ``is None`` check and
# each constructing their own pool/semaphore (the first one leaks).
_async_pool_inst: Optional[concurrent.futures.ThreadPoolExecutor] = None
_async_sem_inst: Optional[threading.BoundedSemaphore] = None
_async_init_lock = threading.Lock()
# ``_async_shutting_down`` gates new work: signal handlers flip it
# early so in-flight workers demote their exit-code log to DEBUG
# (a signal-delivered SIGTERM/SIGKILL on the subprocess is expected,
# not an error) and ``_spawn`` self-terminates after the ``_live_procs``
# snapshot.  Cleanup idempotency lives on ``_async_cleanup_ran`` below
# so that ``shutdown_async_hooks`` still runs the SIGTERM-wait-SIGKILL
# escalation when the signal handler flipped the shutting-down flag
# *before* the atexit hook / gateway path got to call us.
_async_shutting_down = False
_async_cleanup_lock = threading.Lock()
_async_cleanup_ran = False
_async_atexit_registered = False

# Signal chaining state: only CLI-mode installs handlers (the gateway
# owns its own asyncio-native handlers, see gateway/run.py).
# ``_signal_handlers_owned_by_us`` is true only after
# ``_maybe_install_signal_handlers`` actually swapped them — tests that
# set ``_sigint_handler_installed`` manually don't flip it, so
# ``_reset_async_pool`` won't restore a handler the module never owned
# back onto the test runner's signal disposition.
_original_sigint_handler: Any = None
_original_sigterm_handler: Any = None
_sigint_handler_installed = False
_sigint_handler_owned_by_us = False


@dataclass
class ShellHookSpec:
    """Parsed and validated representation of a single ``hooks:`` entry."""

    event: str
    command: str
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    is_async: bool = False
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Strip whitespace introduced by YAML quirks (e.g. multi-line string
        # folding) — a matcher of " terminal" would otherwise silently fail
        # to match "terminal" without any diagnostic.
        if isinstance(self.matcher, str):
            stripped = self.matcher.strip()
            self.matcher = stripped if stripped else None
        if self.matcher:
            try:
                self.compiled_matcher = re.compile(self.matcher)
            except re.error as exc:
                logger.warning(
                    "shell hook matcher %r is invalid (%s) — treating as "
                    "literal equality", self.matcher, exc,
                )
                self.compiled_matcher = None

    def matches_tool(self, tool_name: Optional[str]) -> bool:
        if not self.matcher:
            return True
        if tool_name is None:
            return False
        if self.compiled_matcher is not None:
            return self.compiled_matcher.fullmatch(tool_name) is not None
        # compiled_matcher is None only when the regex failed to compile,
        # in which case we already warned and fall back to literal equality.
        return tool_name == self.matcher


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_from_config(
    cfg: Optional[Dict[str, Any]],
    *,
    accept_hooks: bool = False,
) -> List[ShellHookSpec]:
    """Register every configured shell hook on the plugin manager.

    ``cfg`` is the full parsed config dict (``hermes_cli.config.load_config``
    output).  The ``hooks:`` key is read out of it.  Missing, empty, or
    non-dict ``hooks`` is treated as zero configured hooks.

    ``accept_hooks=True`` skips the TTY consent prompt — the caller is
    promising that the user has opted in via a flag, env var, or config
    setting.  ``HERMES_ACCEPT_HOOKS=1`` and ``hooks_auto_accept: true`` are
    also honored inside this function so either CLI or gateway call sites
    pick them up.

    Returns the list of :class:`ShellHookSpec` entries that ended up wired
    up on the plugin manager.  Skipped entries (unknown events, malformed,
    not allowlisted, already registered) are logged but not returned.
    """
    if not isinstance(cfg, dict):
        return []

    effective_accept = _resolve_effective_accept(cfg, accept_hooks)

    specs = _parse_hooks_block(cfg.get("hooks"))
    if not specs:
        return []

    registered: List[ShellHookSpec] = []

    # Import lazily — avoids circular imports at module-load time.
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()

    # Idempotence + allowlist read happen under the lock; the TTY
    # prompt runs outside so other threads aren't parked on a blocking
    # input().  Mutation re-takes the lock with a defensive idempotence
    # re-check in case two callers ever race through the prompt.
    for spec in specs:
        key = (spec.event, spec.matcher, spec.command)
        with _registered_lock:
            if key in _registered:
                continue
            already_allowlisted = _is_allowlisted(spec.event, spec.command)

        if not already_allowlisted:
            if not _prompt_and_record(
                spec.event, spec.command, accept_hooks=effective_accept,
            ):
                logger.warning(
                    "shell hook for %s (%s) not allowlisted — skipped. "
                    "Use --accept-hooks / HERMES_ACCEPT_HOOKS=1 / "
                    "hooks_auto_accept: true, or approve at the TTY "
                    "prompt next run.",
                    spec.event, spec.command,
                )
                continue

        with _registered_lock:
            if key in _registered:
                continue
            manager._hooks.setdefault(spec.event, []).append(_make_callback(spec))
            _registered.add(key)
            registered.append(spec)
            logger.info(
                "shell hook registered: %s -> %s (matcher=%s, timeout=%ds)",
                spec.event, spec.command, spec.matcher, spec.timeout,
            )

    return registered


def iter_configured_hooks(cfg: Optional[Dict[str, Any]]) -> List[ShellHookSpec]:
    """Return the parsed ``ShellHookSpec`` entries from config without
    registering anything.  Used by ``hermes hooks list`` and ``doctor``."""
    if not isinstance(cfg, dict):
        return []
    return _parse_hooks_block(cfg.get("hooks"))


def reset_for_tests() -> None:
    """Clear the idempotence set.  Test-only helper."""
    with _registered_lock:
        _registered.clear()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_hooks_block(hooks_cfg: Any) -> List[ShellHookSpec]:
    """Normalise the ``hooks:`` dict into a flat list of ``ShellHookSpec``.

    Malformed entries warn-and-skip — we never raise from config parsing
    because a broken hook must not crash the agent.
    """
    from hermes_cli.plugins import VALID_HOOKS

    if not isinstance(hooks_cfg, dict):
        return []

    specs: List[ShellHookSpec] = []

    for event_name, entries in hooks_cfg.items():
        if event_name not in VALID_HOOKS:
            suggestion = difflib.get_close_matches(
                str(event_name), VALID_HOOKS, n=1, cutoff=0.6,
            )
            if suggestion:
                logger.warning(
                    "unknown hook event %r in hooks: config — did you mean %r?",
                    event_name, suggestion[0],
                )
            else:
                logger.warning(
                    "unknown hook event %r in hooks: config (valid: %s)",
                    event_name, ", ".join(sorted(VALID_HOOKS)),
                )
            continue

        if entries is None:
            continue

        if not isinstance(entries, list):
            logger.warning(
                "hooks.%s must be a list of hook definitions; got %s",
                event_name, type(entries).__name__,
            )
            continue

        for i, raw in enumerate(entries):
            spec = _parse_single_entry(event_name, i, raw)
            if spec is not None:
                specs.append(spec)

    return specs


def _parse_single_entry(
    event: str, index: int, raw: Any,
) -> Optional[ShellHookSpec]:
    if not isinstance(raw, dict):
        logger.warning(
            "hooks.%s[%d] must be a mapping with a 'command' key; got %s",
            event, index, type(raw).__name__,
        )
        return None

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        logger.warning(
            "hooks.%s[%d] is missing a non-empty 'command' field",
            event, index,
        )
        return None

    matcher = raw.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        logger.warning(
            "hooks.%s[%d].matcher must be a string regex; ignoring",
            event, index,
        )
        matcher = None

    if matcher is not None and event not in ("pre_tool_call", "post_tool_call"):
        logger.warning(
            "hooks.%s[%d].matcher=%r will be ignored at runtime — the "
            "matcher field is only honored for pre_tool_call / "
            "post_tool_call.  The hook will fire on every %s event.",
            event, index, matcher, event,
        )
        matcher = None

    timeout_raw = raw.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        logger.warning(
            "hooks.%s[%d].timeout must be an int (got %r); using default %ds",
            event, index, timeout_raw, DEFAULT_TIMEOUT_SECONDS,
        )
        timeout = DEFAULT_TIMEOUT_SECONDS

    if timeout < 1:
        logger.warning(
            "hooks.%s[%d].timeout must be >=1; using default %ds",
            event, index, DEFAULT_TIMEOUT_SECONDS,
        )
        timeout = DEFAULT_TIMEOUT_SECONDS

    if timeout > MAX_TIMEOUT_SECONDS:
        logger.warning(
            "hooks.%s[%d].timeout=%ds exceeds max %ds; clamping",
            event, index, timeout, MAX_TIMEOUT_SECONDS,
        )
        timeout = MAX_TIMEOUT_SECONDS

    async_raw = raw.get("async", False)
    if not isinstance(async_raw, bool):
        logger.warning(
            "hooks.%s[%d].async must be a boolean; got %r. Defaulting to sync.",
            event, index, async_raw,
        )
        is_async = False
    else:
        is_async = async_raw

    if is_async:
        if event in _REJECT_ASYNC_EVENTS:
            logger.error(
                "hooks.%s[%d] cannot use async: true — the %s event only "
                "propagates through its return value (context injection / "
                "tool-result or terminal-output transform), and an async "
                "firing would discard it. Skipping this hook.",
                event, index, event,
            )
            return None
        if event in _WARN_ASYNC_EVENTS:
            logger.warning(
                "hooks.%s[%d] uses async: true — %s block directives will "
                "be ignored because the tool has already executed by the "
                "time the background subprocess completes. Accepted for "
                "observational use cases.",
                event, index, event,
            )

    return ShellHookSpec(
        event=event,
        command=command.strip(),
        matcher=matcher,
        timeout=timeout,
        is_async=is_async,
    )


# ---------------------------------------------------------------------------
# Subprocess callback
# ---------------------------------------------------------------------------

_TOP_LEVEL_PAYLOAD_KEYS = {"tool_name", "args", "session_id", "parent_session_id"}


def _register_live_proc(proc: subprocess.Popen) -> None:
    with _async_tracking_lock:
        _live_procs.add(proc)


def _unregister_live_proc(proc: subprocess.Popen) -> None:
    with _async_tracking_lock:
        _live_procs.discard(proc)


def _terminate_group(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the subprocess's entire process group.

    On POSIX, hooks are spawned with ``start_new_session=True`` so the
    subprocess is the leader of a fresh PGID; orphaned grandchildren
    (e.g. a bash script's ``sleep``) would otherwise keep the parent
    stdout FD open and block ``proc.communicate``.  Sending SIGTERM to
    the group cascades to every descendant.  Falls back to a plain
    ``terminate`` on edge cases where the group lookup fails (e.g. the
    proc already exited).

    On Windows the POSIX process-group APIs raise ``AttributeError``
    (not ``OSError``) and ``start_new_session`` is silently ignored, so
    the best we can do is single-process ``proc.terminate()``."""
    if _IS_WINDOWS:
        try:
            proc.terminate()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def _kill_group(proc: subprocess.Popen) -> None:
    """Same as :func:`_terminate_group` but with SIGKILL, for
    subprocesses that didn't honor SIGTERM within the grace window.
    ``signal.SIGKILL`` does not exist on Windows — ``proc.kill()`` is
    the platform analogue."""
    if _IS_WINDOWS:
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _register_future(fut: concurrent.futures.Future) -> None:
    with _async_tracking_lock:
        _live_futures.add(fut)


def _unregister_future(fut: concurrent.futures.Future) -> None:
    with _async_tracking_lock:
        _live_futures.discard(fut)


# ---------------------------------------------------------------------------
# Async config lookups (clamped on read — cheap enough to do per submit)
# ---------------------------------------------------------------------------

def _clamp_int(value: Any, default: int, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n < lo:
        logger.warning(
            "async-hook config value %r below minimum %d; clamping", n, lo,
        )
        return lo
    if n > hi:
        logger.warning(
            "async-hook config value %r above maximum %d; clamping", n, hi,
        )
        return hi
    return n


def _async_config() -> Tuple[int, int]:
    """Return ``(pool_size, shutdown_grace_seconds)`` from config.

    Read lazily so tests can monkeypatch ``load_config`` before the pool
    is first materialised.  Failure falls back to the module defaults.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    pool = _clamp_int(
        cfg.get("hooks_async_pool_size"),
        _DEFAULT_ASYNC_POOL_SIZE,
        _ASYNC_POOL_SIZE_BOUNDS,
    )
    grace = _clamp_int(
        cfg.get("hooks_async_shutdown_grace_seconds"),
        _DEFAULT_ASYNC_SHUTDOWN_GRACE,
        _ASYNC_SHUTDOWN_GRACE_BOUNDS,
    )
    return pool, grace


def _async_pool_get() -> concurrent.futures.ThreadPoolExecutor:
    """Lazy-init the async pool.  Idempotent across CLI + gateway
    startup; registers the atexit shutdown sweep on first call.
    Double-checked under ``_async_init_lock`` so concurrent callbacks
    can't race into creating two pools."""
    global _async_pool_inst, _async_atexit_registered
    if _async_pool_inst is None:
        with _async_init_lock:
            if _async_pool_inst is None:
                pool_size, _ = _async_config()
                _async_pool_inst = concurrent.futures.ThreadPoolExecutor(
                    max_workers=pool_size,
                    thread_name_prefix="hermes-async-hook",
                )
                if not _async_atexit_registered:
                    atexit.register(shutdown_async_hooks)
                    _async_atexit_registered = True
    return _async_pool_inst


def _async_sem_get() -> threading.BoundedSemaphore:
    """Lazy-init the bounded semaphore that provides real backpressure
    on async hook firings.  Sized to match the pool.  Double-checked
    under the shared init lock."""
    global _async_sem_inst
    if _async_sem_inst is None:
        with _async_init_lock:
            if _async_sem_inst is None:
                pool_size, _ = _async_config()
                _async_sem_inst = threading.BoundedSemaphore(pool_size)
    return _async_sem_inst


def _on_async_future_done(fut: concurrent.futures.Future) -> None:
    """Unified done_callback: release the backpressure semaphore, drop
    tracking, and surface worker-body crashes so they don't vanish.

    Snapshots ``_async_sem_inst`` instead of going through
    ``_async_sem_get`` — if ``_reset_async_pool`` ran between submit
    and callback, the live semaphore is gone and there is nothing to
    release.  Creating a fresh semaphore just to release on it would
    leave it with count > its bound."""
    try:
        exc = fut.exception()
    except concurrent.futures.CancelledError:
        exc = None

    if exc is not None:
        logger.warning("async shell hook worker crashed", exc_info=exc)

    sem = _async_sem_inst
    if sem is not None:
        try:
            sem.release()
        except ValueError:
            # Over-release only happens if the semaphore was swapped
            # out mid-flight (e.g. test reset).  Benign; ignore.
            logger.debug("async semaphore over-release; ignored")
    _unregister_future(fut)


def _run_async_body(spec: ShellHookSpec, payload: str) -> None:
    """Worker body: run the hook, log the outcome, discard the response.

    Block directives on ``pre_tool_call`` are explicitly logged at WARN
    so users who migrate a sync ``pre_tool_call`` hook to async don't
    silently lose their block.
    """
    r = _spawn(spec, payload)

    if r.get("error"):
        logger.warning(
            "async shell hook failed (event=%s command=%s): %s",
            spec.event, spec.command, r["error"],
        )
        return
    if r.get("timed_out"):
        logger.warning(
            "async shell hook timed out after %.2fs (event=%s command=%s)",
            r.get("elapsed_seconds", 0), spec.event, spec.command,
        )
        return

    stderr = (r.get("stderr") or "").strip()
    if stderr:
        logger.debug(
            "async shell hook stderr (event=%s command=%s): %s",
            spec.event, spec.command, stderr[:400],
        )
    rc = r.get("returncode")
    # During shutdown we SIGTERM every tracked child ourselves, so a
    # negative return code (-SIGTERM = -15, -SIGKILL = -9) is expected
    # and not a hook-author failure.  Demote to DEBUG in that case so
    # the normal shutdown path doesn't emit a pool-sized burst of WARN.
    if rc not in (0, None):
        level = (
            logging.DEBUG
            if _async_shutting_down and rc is not None and rc < 0
            else logging.WARNING
        )
        logger.log(
            level,
            "async shell hook exited %d (event=%s command=%s); stderr=%s",
            rc, spec.event, spec.command, stderr[:400],
        )

    parsed = _parse_response(spec.event, r.get("stdout") or "")
    if (
        spec.event == "pre_tool_call"
        and parsed
        and parsed.get("action") == "block"
    ):
        logger.warning(
            "async shell hook returned block directive but firing is "
            "async — tool has already executed (event=%s command=%s "
            "message=%r)",
            spec.event, spec.command, parsed.get("message"),
        )


def shutdown_async_hooks(grace_seconds: Optional[float] = None) -> None:
    """Tear down the async hook pool.

    Safe to call from:
      * the atexit handler (CLI default)
      * the gateway's asyncio shutdown path
      * ``_reset_async_pool()`` in tests

    Idempotent — subsequent calls after the first are no-ops.
    Terminates every tracked subprocess with SIGTERM, waits up to
    ``grace_seconds`` (default: config-derived) for clean exits, and
    SIGKILLs survivors.  Workers exit once their subprocess is gone.

    The idempotency guard lives on ``_async_cleanup_ran`` rather than
    ``_async_shutting_down`` so the signal-handler path still benefits
    from the escalation: ``_async_pool_sig{int,term}_handler`` flip
    ``_async_shutting_down`` before atexit fires us, so gating on that
    flag would skip the SIGKILL cascade whenever a hook script ignores
    SIGTERM.
    """
    global _async_shutting_down, _async_cleanup_ran
    with _async_cleanup_lock:
        if _async_cleanup_ran:
            return
        _async_cleanup_ran = True
    _async_shutting_down = True

    pool = _async_pool_inst
    if pool is not None:
        pool.shutdown(wait=False)

    if grace_seconds is None:
        _, grace_seconds = _async_config()

    with _async_tracking_lock:
        procs = list(_live_procs)
    for proc in procs:
        _terminate_group(proc)

    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            logger.warning(
                "async shell hook subprocess exceeded shutdown grace "
                "(%.1fs) — killed pid=%d",
                grace_seconds, proc.pid,
            )

    if pool is not None:
        try:
            pool.shutdown(wait=True)
        except Exception:
            pass


def _async_pool_sigint_handler(signum, frame):
    """CLI-mode SIGINT handler: run the full async-hook shutdown
    inline (SIGTERM → wait grace_seconds → SIGKILL survivors), then
    chain to the original handler so ``KeyboardInterrupt`` propagation
    still happens.

    Inline shutdown — rather than the previous SIGTERM-only cascade
    that relied on atexit for the SIGKILL escalation — is required
    because ``ThreadPoolExecutor`` uses non-daemon threads: the
    interpreter waits for every worker to return *before* running
    atexit, and workers block in ``proc.communicate(timeout=spec.timeout)``
    until the subprocess dies.  A hook script that ignores SIGTERM
    would therefore delay Ctrl-C by ``spec.timeout`` (60 s default,
    300 s max) instead of the intended ``grace_seconds``.  Running
    ``shutdown_async_hooks`` here drops that to ``grace_seconds``.

    Installed only by CLI-mode callers (see
    ``_maybe_install_signal_handlers``); the gateway integrates
    ``shutdown_async_hooks`` into its asyncio shutdown sequence
    instead.
    """
    try:
        shutdown_async_hooks()
    except Exception:
        # Signal handlers must never raise out of cleanup — the chain
        # / KeyboardInterrupt logic below is what the user actually
        # expects to see.
        pass

    handler = _original_sigint_handler
    if handler in (signal.SIG_DFL, signal.SIG_IGN, None):
        raise KeyboardInterrupt()
    handler(signum, frame)


def _async_pool_sigterm_handler(signum, frame):
    """CLI-mode SIGTERM handler: run the full async-hook shutdown
    inline (SIGTERM → wait grace_seconds → SIGKILL survivors), then
    ``sys.exit`` so Python still runs the atexit chain.

    Same reasoning as :func:`_async_pool_sigint_handler` applies: the
    ``ThreadPoolExecutor`` uses non-daemon threads, so the interpreter
    waits for every worker to return before atexit runs.  Unless we
    escalate to SIGKILL here, a TERM-ignoring hook script blocks the
    interpreter for ``spec.timeout`` (60 s default, 300 s max) because
    its worker is stuck in ``proc.communicate``.

    Chains to the previous SIGTERM handler if one was installed.
    Gateways install their own SIGTERM handler on the asyncio loop,
    so this path is CLI-only.
    """
    try:
        shutdown_async_hooks()
    except Exception:
        pass

    handler = _original_sigterm_handler
    if handler not in (signal.SIG_DFL, signal.SIG_IGN, None):
        handler(signum, frame)
    # SystemExit runs the atexit chain, which sees
    # ``_async_cleanup_ran=True`` and no-ops.  128 + signal number
    # matches shell convention for signal-initiated exits.
    sys.exit(128 + signal.SIGTERM)


def _maybe_install_signal_handlers() -> None:
    """Install the CLI async-hook SIGINT + SIGTERM handlers once.

    Only the CLI entry point calls this; the gateway owns its own
    asyncio-native signal disposition.  Wrapped in try/except so
    non-main-thread callers degrade to atexit-only (``signal.signal``
    raises ``ValueError`` off the main thread).

    The SIGINT handler terminates tracked subprocesses and chains to
    the original handler so ``KeyboardInterrupt`` propagation works.
    The SIGTERM handler calls ``sys.exit`` so Python's atexit chain
    runs — without this, SIGTERM (from ``kill``, ``timeout``, systemd
    stop, CI harnesses) skips atexit and orphans every in-flight hook
    subprocess.
    """
    global _original_sigint_handler, _original_sigterm_handler
    global _sigint_handler_installed, _sigint_handler_owned_by_us
    if _sigint_handler_installed:
        return
    try:
        _original_sigint_handler = signal.signal(
            signal.SIGINT, _async_pool_sigint_handler,
        )
        _original_sigterm_handler = signal.signal(
            signal.SIGTERM, _async_pool_sigterm_handler,
        )
        _sigint_handler_installed = True
        _sigint_handler_owned_by_us = True
    except (ValueError, OSError):
        logger.debug(
            "signal handlers not installed for async hooks (not main "
            "thread); relying on atexit shutdown + "
            "hooks_async_shutdown_grace_seconds",
        )


# Back-compat alias for the old name (kept so any out-of-tree caller
# that imported it still works; the CLI has been switched to the new
# name in the same commit).
_maybe_install_sigint_handler = _maybe_install_signal_handlers


def _reset_async_pool() -> None:
    """Test-only: drop every piece of async-pool module state.

    Complements :func:`reset_for_tests` (which clears the registration
    set).  Must be called from test teardown that varied pool size,
    grace window, or triggered shutdown during the test; otherwise the
    next test sees stale singletons and leaked subprocesses.
    """
    global _async_pool_inst, _async_sem_inst, _async_shutting_down
    global _async_cleanup_ran
    global _original_sigint_handler, _original_sigterm_handler
    global _sigint_handler_installed, _sigint_handler_owned_by_us

    if _async_pool_inst is not None or _live_procs:
        try:
            shutdown_async_hooks(grace_seconds=0.5)
        except Exception:
            pass

    with _async_tracking_lock:
        _live_procs.clear()
        _live_futures.clear()

    _async_pool_inst = None
    _async_sem_inst = None
    _async_shutting_down = False
    _async_cleanup_ran = False

    # Only restore signal handlers we actually installed ourselves —
    # tests that flip ``_sigint_handler_installed`` manually (to
    # exercise ``_async_pool_sigint_handler`` without running
    # ``_maybe_install_signal_handlers``) haven't mutated the
    # process's real signal disposition, so we must not write a fake
    # handler back into it.
    if _sigint_handler_owned_by_us:
        for sig, previous in (
            (signal.SIGINT, _original_sigint_handler),
            (signal.SIGTERM, _original_sigterm_handler),
        ):
            if previous is None:
                continue
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
    _original_sigint_handler = None
    _original_sigterm_handler = None
    _sigint_handler_installed = False
    _sigint_handler_owned_by_us = False


def _spawn(spec: ShellHookSpec, stdin_json: str) -> Dict[str, Any]:
    """Run ``spec.command`` as a subprocess with ``stdin_json`` on stdin.

    Returns a diagnostic dict with the same keys for every outcome
    (``returncode``, ``stdout``, ``stderr``, ``timed_out``,
    ``elapsed_seconds``, ``error``).  This is the single place the
    subprocess is actually invoked — both the live callback path
    (:func:`_make_callback`) and the CLI test helper (:func:`run_once`)
    go through it.

    Uses :class:`subprocess.Popen` + :meth:`Popen.communicate` with the
    handle registered in :data:`_live_procs` for the duration of the
    call.  That lets the async-pool shutdown path and SIGINT handler
    terminate the child externally, without waiting for the full
    ``spec.timeout`` to elapse inside a worker thread.  For sync
    callers the behaviour is semantically identical to the prior
    ``subprocess.run`` implementation (block until exit or timeout,
    terminate+kill on timeout).
    """
    result: Dict[str, Any] = {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "error": None,
    }
    try:
        argv = shlex.split(os.path.expanduser(spec.command))
    except ValueError as exc:
        result["error"] = f"command {spec.command!r} cannot be parsed: {exc}"
        return result
    if not argv:
        result["error"] = "empty command"
        return result

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            # Give the subprocess its own PGID so shutdown can kill
            # the whole tree — otherwise orphaned grandchildren (a
            # bash script's ``sleep``, a python script's
            # ``multiprocessing`` workers, ...) keep the parent
            # stdout FD open and ``proc.communicate`` blocks until
            # they exit on their own.
            start_new_session=True,
        )
    except FileNotFoundError:
        result["error"] = "command not found"
        return result
    except PermissionError:
        result["error"] = "command not executable"
        return result
    except OSError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:  # pragma: no cover — defensive
        result["error"] = str(exc)
        return result

    _register_live_proc(proc)
    # Close the shutdown race: if shutdown_async_hooks() snapshotted
    # _live_procs *before* we registered this proc, it will not have
    # been SIGTERM'd and proc.communicate() below would block for up
    # to spec.timeout.  Self-terminate so the worker exits promptly.
    if _async_shutting_down:
        _terminate_group(proc)
    try:
        try:
            stdout, stderr = proc.communicate(
                input=stdin_json, timeout=spec.timeout,
            )
        except subprocess.TimeoutExpired:
            # Terminate the stuck child and drain any partial output so
            # the caller doesn't deadlock on pipe buffers.
            _terminate_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except Exception:
                    stdout, stderr = "", ""
            result["timed_out"] = True
            result["stdout"] = stdout or ""
            result["stderr"] = stderr or ""
            result["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            return result
    finally:
        _unregister_live_proc(proc)

    result["returncode"] = proc.returncode
    result["stdout"] = stdout or ""
    result["stderr"] = stderr or ""
    result["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    return result


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[Dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing.

    Dispatches to :func:`_make_async_callback` for ``async: true``
    specs; otherwise returns the sync callback (identical behaviour to
    the pre-async path).
    """
    if spec.is_async:
        return _make_async_callback(spec)
    return _make_sync_callback(spec)


def _make_sync_callback(
    spec: ShellHookSpec,
) -> Callable[..., Optional[Dict[str, Any]]]:
    def _callback(**kwargs: Any) -> Optional[Dict[str, Any]]:
        # Matcher gate — only meaningful for tool-scoped events.
        if spec.event in ("pre_tool_call", "post_tool_call"):
            if not spec.matches_tool(kwargs.get("tool_name")):
                return None

        r = _spawn(spec, _serialize_payload(spec.event, kwargs))

        if r["error"]:
            logger.warning(
                "shell hook failed (event=%s command=%s): %s",
                spec.event, spec.command, r["error"],
            )
            return None
        if r["timed_out"]:
            logger.warning(
                "shell hook timed out after %.2fs (event=%s command=%s)",
                r["elapsed_seconds"], spec.event, spec.command,
            )
            return None

        stderr = r["stderr"].strip()
        if stderr:
            logger.debug(
                "shell hook stderr (event=%s command=%s): %s",
                spec.event, spec.command, stderr[:400],
            )
        # Non-zero exits: log but still parse stdout so scripts that
        # signal failure via exit code can also return a block directive.
        if r["returncode"] != 0:
            logger.warning(
                "shell hook exited %d (event=%s command=%s); stderr=%s",
                r["returncode"], spec.event, spec.command, stderr[:400],
            )

        parsed, extras = _parse_response_with_extras(spec.event, r["stdout"])

        # Claude-Code compatibility: warn-and-ignore if the script
        # declared itself async at runtime.  Hermes only supports
        # config-declared async — this keeps copy-pasted scripts from
        # confusing hook authors.
        if extras.get("async") is True:
            logger.warning(
                "shell hook requested async at runtime but Hermes only "
                "supports config-declared async (event=%s command=%s); "
                "runtime marker discarded.",
                spec.event, spec.command,
            )

        return parsed

    _callback.__name__ = f"shell_hook[{spec.event}:{spec.command}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _make_async_callback(spec: ShellHookSpec) -> Callable[..., None]:
    """Build a fire-and-forget callback that submits the hook to the
    async pool.  Returns immediately; the subprocess runs on a worker
    thread.  Saturation drops the firing with a WARN log.
    """
    def _callback(**kwargs: Any) -> None:
        if spec.event in ("pre_tool_call", "post_tool_call"):
            if not spec.matches_tool(kwargs.get("tool_name")):
                return None

        # Early-exit if the pool is already tearing down.  Without this,
        # a submission that slipped between shutdown-flag-set and
        # pool.shutdown() would spawn a subprocess that misses the
        # termination sweep and delays shutdown via shutdown(wait=True).
        if _async_shutting_down:
            logger.debug(
                "async pool shutting down — dropping firing "
                "event=%s command=%s",
                spec.event, spec.command,
            )
            return None

        payload = _serialize_payload(spec.event, kwargs)

        # Real backpressure: pool is saturated, drop rather than queue
        # unboundedly.  ThreadPoolExecutor's internal queue is
        # unbounded; without this, a chatty post_tool_call could
        # accumulate thousands of queued firings and lose most at
        # shutdown.
        if not _async_sem_get().acquire(blocking=False):
            logger.warning(
                "async shell hook pool saturated — dropping firing "
                "event=%s command=%s",
                spec.event, spec.command,
            )
            return None

        try:
            fut = _async_pool_get().submit(_run_async_body, spec, payload)
        except RuntimeError:
            # Symmetric to _on_async_future_done: snapshot the sem so
            # _reset_async_pool running between acquire and release
            # doesn't prompt us into lazy-creating a fresh sem just to
            # over-release on it.
            sem = _async_sem_inst
            if sem is not None:
                try:
                    sem.release()
                except ValueError:
                    logger.debug("async semaphore over-release; ignored")
            logger.debug(
                "async pool shutting down — dropping firing "
                "event=%s command=%s",
                spec.event, spec.command,
            )
            return None

        _register_future(fut)
        fut.add_done_callback(_on_async_future_done)
        return None

    _callback.__name__ = f"async_shell_hook[{spec.event}:{spec.command}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _serialize_payload(event: str, kwargs: Dict[str, Any]) -> str:
    """Render the stdin JSON payload.  Unserialisable values are
    stringified via ``default=str`` rather than dropped."""
    extras = {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS}
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    payload = {
        "hook_event_name": event,
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        "extra": extras,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _parse_response(event: str, stdout: str) -> Optional[Dict[str, Any]]:
    """Translate stdout JSON into a Hermes wire-shape dict.

    For ``pre_tool_call`` the Claude-Code-style ``{"decision": "block",
    "reason": "..."}`` payload is translated into the canonical Hermes
    ``{"action": "block", "message": "..."}`` shape expected by
    :func:`hermes_cli.plugins.get_pre_tool_call_block_message`.  This is
    the single most important correctness invariant in this module —
    skipping the translation silently breaks every ``pre_tool_call``
    block directive.

    For ``pre_llm_call``, ``{"context": "..."}`` is passed through
    unchanged to match the existing plugin-hook contract.

    Anything else returns ``None``.
    """
    parsed, _ = _parse_response_with_extras(event, stdout)
    return parsed


def _parse_response_with_extras(
    event: str, stdout: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Like :func:`_parse_response` but also returns the raw decoded
    dict so callers can inspect fields that aren't part of the canonical
    wire shape (today: the Claude-Code runtime ``async`` marker).  The
    extras dict is empty whenever the stdout failed to decode as a JSON
    object.  Single parse — used by :func:`_make_sync_callback` so the
    callback doesn't have to JSON-decode the same string twice."""
    stdout = (stdout or "").strip()
    if not stdout:
        return None, {}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning(
            "shell hook stdout was not valid JSON (event=%s): %s",
            event, stdout[:200],
        )
        return None, {}

    if not isinstance(data, dict):
        return None, {}

    if event == "pre_tool_call":
        if data.get("action") == "block":
            message = data.get("message") or data.get("reason") or ""
            if isinstance(message, str) and message:
                return {"action": "block", "message": message}, data
        if data.get("decision") == "block":
            message = data.get("reason") or data.get("message") or ""
            if isinstance(message, str) and message:
                return {"action": "block", "message": message}, data
        return None, data

    context = data.get("context")
    if isinstance(context, str) and context.strip():
        return {"context": context}, data

    return None, data


# ---------------------------------------------------------------------------
# Allowlist / consent
# ---------------------------------------------------------------------------

def allowlist_path() -> Path:
    """Path to the per-user shell-hook allowlist file."""
    return get_hermes_home() / ALLOWLIST_FILENAME


def load_allowlist() -> Dict[str, Any]:
    """Return the parsed allowlist, or an empty skeleton if absent."""
    try:
        raw = json.loads(allowlist_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"approvals": []}
    if not isinstance(raw, dict):
        return {"approvals": []}
    approvals = raw.get("approvals")
    if not isinstance(approvals, list):
        raw["approvals"] = []
    return raw


def save_allowlist(data: Dict[str, Any]) -> None:
    """Atomically persist the allowlist via per-process ``mkstemp`` +
    ``os.replace``.  Cross-process read-modify-write races are handled
    by :func:`_locked_update_approvals` (``fcntl.flock``).  On OSError
    the failure is logged; the in-process hook still registers but
    the approval won't survive across runs."""
    p = allowlist_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{p.name}.", suffix=".tmp", dir=str(p.parent),
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp_path, p)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.warning(
            "Failed to persist shell hook allowlist to %s: %s. "
            "The approval is in-memory for this run, but the next "
            "startup will re-prompt (or skip registration on non-TTY "
            "runs without --accept-hooks / HERMES_ACCEPT_HOOKS).",
            p, exc,
        )


def _is_allowlisted(event: str, command: str) -> bool:
    data = load_allowlist()
    return any(
        isinstance(e, dict)
        and e.get("event") == event
        and e.get("command") == command
        for e in data.get("approvals", [])
    )


@contextmanager
def _locked_update_approvals() -> Iterator[Dict[str, Any]]:
    """Serialise read-modify-write on the allowlist across processes.

    Holds an exclusive ``flock`` on a sibling lock file for the duration
    of the update so concurrent ``_record_approval``/``revoke`` callers
    cannot clobber each other's changes (the race Codex reproduced with
    20–50 simultaneous writers).  Falls back to an in-process lock on
    platforms without ``fcntl``.
    """
    p = allowlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(p.suffix + ".lock")

    if fcntl is None:  # pragma: no cover — non-POSIX fallback
        with _allowlist_write_lock:
            data = load_allowlist()
            yield data
            save_allowlist(data)
        return

    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = load_allowlist()
            yield data
            save_allowlist(data)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _prompt_and_record(
    event: str, command: str, *, accept_hooks: bool,
) -> bool:
    """Decide whether to approve an unseen ``(event, command)`` pair.
    Returns ``True`` iff the approval was granted and recorded.
    """
    if accept_hooks:
        _record_approval(event, command)
        logger.info(
            "shell hook auto-approved via --accept-hooks / env / config: "
            "%s -> %s", event, command,
        )
        return True

    if not sys.stdin.isatty():
        return False

    print(
        f"\n⚠ Hermes is about to register a shell hook that will run a\n"
        f"  command on your behalf.\n\n"
        f"    Event:   {event}\n"
        f"    Command: {command}\n\n"
        f"  Commands run with your full user credentials.  Only approve\n"
        f"  commands you trust."
    )
    try:
        answer = input("Allow this hook to run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # keep the terminal tidy after ^C
        return False

    if answer in ("y", "yes"):
        _record_approval(event, command)
        return True

    return False


def _record_approval(event: str, command: str) -> None:
    entry = {
        "event": event,
        "command": command,
        "approved_at": _utc_now_iso(),
        "script_mtime_at_approval": script_mtime_iso(command),
    }
    with _locked_update_approvals() as data:
        data["approvals"] = [
            e for e in data.get("approvals", [])
            if not (
                isinstance(e, dict)
                and e.get("event") == event
                and e.get("command") == command
            )
        ] + [entry]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def revoke(command: str) -> int:
    """Remove every allowlist entry matching ``command``.

    Returns the number of entries removed.  Does not unregister any
    callbacks that are already live on the plugin manager in the current
    process — restart the CLI / gateway to drop them.
    """
    with _locked_update_approvals() as data:
        before = len(data.get("approvals", []))
        data["approvals"] = [
            e for e in data.get("approvals", [])
            if not (isinstance(e, dict) and e.get("command") == command)
        ]
        after = len(data["approvals"])
    return before - after


_SCRIPT_EXTENSIONS: Tuple[str, ...] = (
    ".sh", ".bash", ".zsh", ".fish",
    ".py", ".pyw",
    ".rb", ".pl", ".lua",
    ".js", ".mjs", ".cjs", ".ts",
)


def _command_script_path(command: str) -> str:
    """Return the script path from ``command`` for doctor / drift checks.

    Prefers a token ending in a known script extension, then a token
    containing ``/`` or leading ``~``, then the first token.  Handles
    ``python3 /path/hook.py``, ``/usr/bin/env bash hook.sh``, and the
    common bare-path form.
    """
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if not parts:
        return command
    for part in parts:
        if part.lower().endswith(_SCRIPT_EXTENSIONS):
            return part
    for part in parts:
        if "/" in part or part.startswith("~"):
            return part
    return parts[0]


# ---------------------------------------------------------------------------
# Helpers for accept-hooks resolution
# ---------------------------------------------------------------------------

def _resolve_effective_accept(
    cfg: Dict[str, Any], accept_hooks_arg: bool,
) -> bool:
    """Combine all three opt-in channels into a single boolean.

    Precedence (any truthy source flips us on):
      1. ``--accept-hooks`` flag (CLI) / explicit argument
      2. ``HERMES_ACCEPT_HOOKS`` env var
      3. ``hooks_auto_accept: true`` in ``cli-config.yaml``
    """
    if accept_hooks_arg:
        return True
    env = os.environ.get("HERMES_ACCEPT_HOOKS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    cfg_val = cfg.get("hooks_auto_accept", False)
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        return cfg_val.strip().lower() in ("1", "true", "yes", "on")
    return False


# ---------------------------------------------------------------------------
# Introspection (used by `hermes hooks` CLI)
# ---------------------------------------------------------------------------

def allowlist_entry_for(event: str, command: str) -> Optional[Dict[str, Any]]:
    """Return the allowlist record for this pair, if any."""
    for e in load_allowlist().get("approvals", []):
        if (
            isinstance(e, dict)
            and e.get("event") == event
            and e.get("command") == command
        ):
            return e
    return None


def script_mtime_iso(command: str) -> Optional[str]:
    """ISO-8601 mtime of the resolved script path, or ``None`` if the
    script is missing."""
    path = _command_script_path(command)
    if not path:
        return None
    try:
        expanded = os.path.expanduser(path)
        return datetime.fromtimestamp(
            os.path.getmtime(expanded), tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def script_is_executable(command: str) -> bool:
    """Return ``True`` iff ``command`` is runnable as configured.

    For a bare invocation (``/path/hook.sh``) the script itself must be
    executable.  For interpreter-prefixed commands (``python3
    /path/hook.py``, ``/usr/bin/env bash hook.sh``) the script just has
    to be readable — the interpreter doesn't care about the ``X_OK``
    bit.  Mirrors what ``_spawn`` would actually do at runtime."""
    path = _command_script_path(command)
    if not path:
        return False
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    is_bare_invocation = bool(argv) and argv[0] == path
    required = os.X_OK if is_bare_invocation else os.R_OK
    return os.access(expanded, required)


def run_once(
    spec: ShellHookSpec, kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Fire a single shell-hook invocation with a synthetic payload.
    Used by ``hermes hooks test`` and ``hermes hooks doctor``.

    ``kwargs`` is the same dict that :func:`hermes_cli.plugins.invoke_hook`
    would pass at runtime.  It is routed through :func:`_serialize_payload`
    so the synthetic stdin exactly matches what a real hook firing would
    produce — otherwise scripts tested via ``hermes hooks test`` could
    diverge silently from production behaviour.

    Returns the :func:`_spawn` diagnostic dict plus a ``parsed`` field
    holding the canonical Hermes-wire-shape response."""
    stdin_json = _serialize_payload(spec.event, kwargs)
    result = _spawn(spec, stdin_json)
    result["parsed"] = _parse_response(spec.event, result["stdout"])
    return result
